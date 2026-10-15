"""Serial feature refresh, validated Grounding, and lambda experiments on LAB."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import yaml

DATASETS = {'redial':'hyprorec-redial', 'tg_redial':'hyprorec-tg-redial'}


def save_json(path, value):
    temporary = Path(str(path) + '.partial')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def run_stage(name, command, project, state_path, poll_seconds, retry_failed=False, log_directory=None):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else {'stages':{}}
    stages = state.setdefault('stages', {})
    entry = stages.get(name)
    log_directory = Path(log_directory) if log_directory else state_path.parent / name
    log_directory.mkdir(parents=True,exist_ok=True)
    exit_path = log_directory / 'exit-code.txt'
    if entry is not None and entry.get('exit_code') not in (None,0) and retry_failed:
        if entry['command'] != command:
            raise ValueError(f'Cannot resume changed stage command: {name}')
        history=state.setdefault('failed_attempts',[])
        history.append(dict(entry,stage=name))
        if exit_path.exists():
            exit_path.rename(log_directory/f'exit-code.failed-{len(history)}.txt')
        stages.pop(name)
        entry=None
        save_json(state_path,state)
    if entry is not None:
        if entry['command'] != command:
            raise ValueError(f'Cannot resume changed stage command: {name}')
        if entry.get('exit_code') == 0:
            return
        if exit_path.exists():
            code = int(exit_path.read_text())
        elif entry.get('pid') and Path(f"/proc/{entry['pid']}").exists():
            while not exit_path.exists():
                time.sleep(poll_seconds)
                if not Path(f"/proc/{entry['pid']}").exists() and not exit_path.exists():
                    raise RuntimeError(f'Lost stage exit status: {name}')
            code = int(exit_path.read_text())
        else:
            raise RuntimeError(f'Stage requires explicit recovery: {name}')
    else:
        environment = os.environ.copy()
        environment['HYPROREC_EXIT_FILE'] = str(exit_path.resolve())
        shell = '"$@"; status=$?; printf "%s\\n" "$status" > "$HYPROREC_EXIT_FILE"; exit "$status"'
        with (log_directory / 'console.log').open('a') as log:
            process = subprocess.Popen(['bash','-c',shell,'hyprorec-stage',*command],
                                       cwd=project, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        entry = {'command':command,'pid':process.pid,'started_at':datetime.now(timezone.utc).isoformat()}
        stages[name] = entry
        state['active_stage'] = name
        save_json(state_path, state)
        print(f'START {name} pid={process.pid}', flush=True)
        while process.poll() is None:
            time.sleep(poll_seconds)
        code = process.returncode
    entry['exit_code'] = code
    entry['finished_at'] = datetime.now(timezone.utc).isoformat()
    state['active_stage'] = None
    save_json(state_path, state)
    print(f'END {name} exit={code}', flush=True)
    if code:
        raise subprocess.CalledProcessError(code, command)


def update_configs(project, version, groundings=None):
    for dataset, folder in DATASETS.items():
        for path in (project / 'configs' / dataset).glob('*.yaml'):
            text = path.read_text()
            payload = yaml.safe_load(text)
            root = payload['data']['dataset_path']
            replacements = {
                'embeddings_dir_name':f'feature_versions/{version}/embeddings',
                'hyperedge_table_path':f'{root}/feature_versions/{version}/hyperedge_table.json'}
            if groundings and dataset in groundings and 'grounding_checkpoint_path' in payload['model']:
                replacements['grounding_checkpoint_path'] = groundings[dataset]
            for key, value in replacements.items():
                text, count = re.subn(rf'^(  {key}:\s*).*$', lambda m:m[1]+value, text, flags=re.MULTILINE)
                if count != 1:
                    raise ValueError(f'Expected one editable {key} field in {path}')
            path.write_text(text)


def clean_retired(manifest, allowed_root):
    root = allowed_root.resolve()
    if Path(manifest['root']).resolve() != root:
        raise ValueError('Checkpoint cleanup root differs from worktree outputs')
    targets = [Path(entry['path']) for entry in manifest['targets']]
    for path in targets:
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError('Checkpoint cleanup escaped the frozen whitelist')
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def validate_version(root,dataset):
    manifest=json.loads((root/'feature_manifest.json').read_text())
    if not manifest['validated']:
        raise ValueError('Feature version has not passed validation')
    files={str(path.relative_to(root)):path for path in root.rglob('*')
           if path.is_file() and path.name!='feature_manifest.json'}
    if set(files)!=set(manifest['hashes']):
        raise ValueError('Published feature file set changed')
    for name,digest in manifest['hashes'].items():
        if sha256(files[name])!=digest:
            raise ValueError(f'Published feature changed: {name}')
    for name,key in [('movies_info.csv','csv_sha256'),('alignment_manifest.json','alignment_manifest_sha256')]:
        if sha256(dataset/name)!=manifest[key]:
            raise ValueError('Feature version does not match the canonical dataset')


def gpu_idle():
    output = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True)
    values = [tuple(map(int, row.split(','))) for row in output.splitlines()]
    return len(values) == 8 and all(memory < 100 and utilization == 0 for memory,utilization in values)


def wait_for_gpus(interval):
    while not gpu_idle():
        print('Waiting for all eight GPUs to be idle', flush=True)
        time.sleep(interval)


def prepare_recipe(project, dataset, destination, batch_size=8):
    recipe = yaml.safe_load((project / 'configs' / dataset / 'grounding.yaml').read_text())
    recipe['training'].update({
        'output_dir':str(destination.relative_to(project)),
        'per_device_train_batch_size':batch_size,
        'per_device_eval_batch_size':batch_size,
        'logging_dir':str((destination/'trainer-logs').relative_to(project)),
        'dataloader_num_workers':4,'dataloader_pin_memory':False,
        'dataloader_persistent_workers':False,'dataloader_prefetch_factor':1,
        'gradient_checkpointing':False})
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'run_config.yaml'
    if not path.exists():path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    return path


def check_grounding(path, feature_root):
    config=json.loads((path/'config.json').read_text())
    manifest=json.loads((feature_root/'feature_manifest.json').read_text())
    dimensions={view:check['shape'][1] for view,check in manifest['features'].items()}
    if set(config['views']) != set(dimensions) or config['input_dims'] != dimensions:
        raise ValueError('Grounding checkpoint does not match this feature version')
    if not (path/'model.safetensors').is_file():
        raise FileNotFoundError(path/'model.safetensors')
    best=json.loads((path/'best_modalities.json').read_text())
    if set(best) != set(dimensions) or any(value['eval_loss'] is None for value in best.values()):
        raise ValueError('Every modality must have a selected validation checkpoint')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--poll-seconds',type=float,default=1800)
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    state_path=args.state.resolve()
    stamp=state_path.parent.name
    version='clap-xclip-'+stamp
    python=str(project/'.venv/bin/python')
    data_root=Path('/data2/lhf/.DATA')
    os.environ.update({'PYTHONPATH':str(project/'src'),'TOKENIZERS_PARALLELISM':'false',
                      'OMP_NUM_THREADS':'1','HF_HUB_OFFLINE':'1',
                      'TMPDIR':'/data2/lhf/tmp/hyprorec-refresh'})
    feature_paths={}
    for dataset,folder in DATASETS.items():
        root=data_root/folder
        final=root/'feature_versions'/version
        stage=final.with_name(final.name+'.partial')
        feature_paths[dataset]=final
        if final.exists():
            validate_version(final,root)
            continue
        for phase in ['probe','encode','tables','publish']:
            wait_for_gpus(30)
            os.environ['CUDA_VISIBLE_DEVICES']='0' if phase in ['probe','encode'] else ''
            command=[python,'-u',str(root/'scripts/refresh_features.py'),
                     '--dataset-dir',str(root),'--output-dir',str(stage),'--phase',phase,
                     '--audit-dir',str(project/'outputs'/dataset/'data-audit/20261013T095341826155Z')]
            run_stage(dataset+'-'+phase,command,project,state_path,5,log_directory=project/'outputs'/dataset/'feature-refresh'/stamp/phase)
    for dataset,folder in DATASETS.items():validate_version(feature_paths[dataset],data_root/folder)
    update_configs(project,version)
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    os.environ['SWANLAB_MODE']='disabled'
    torchrun=[python,'-m','torch.distributed.run','--standalone','--nproc-per-node=8']
    smoke_roots={}
    batch_sizes={}
    for dataset in DATASETS:
        wait_for_gpus(30)
        smoke=project/'outputs'/dataset/'refresh-smoke'/stamp
        smoke_roots[dataset]=smoke
        for batch_size in [8,4,2,1]:
            attempt=smoke/('batch-'+str(batch_size))
            name=dataset+'-smoke-b'+str(batch_size)
            command=['timeout','1200s',*torchrun,str(project/'scripts/smoke_refresh.py'),
                     '--dataset',dataset,'--batch-size',str(batch_size),'--output',str(attempt)]
            try:
                run_stage(name,command,project,state_path,5,log_directory=attempt)
                batch_sizes[dataset]=batch_size
                smoke=attempt
                break
            except subprocess.CalledProcessError:
                log=(attempt/'console.log').read_text()
                if batch_size==1 or not ('CUDA out of memory' in log or 'OutOfMemoryError' in log):
                    raise
                print(f'{dataset}: next smoke uses smaller Grounding batch',flush=True)
        smoke_roots[dataset]=smoke
        result=json.loads((smoke/'smoke.json').read_text())
        if not result['passed'] or result['world_size']!=8:
            raise ValueError('All eight ranks must pass Grounding and CRS smoke')
    cleanup_marker=state_path.parent/'cleanup-complete.json'
    if not cleanup_marker.exists():
        manifest=json.loads((state_path.parent/'retired-checkpoints.json').read_text())
        clean_retired(manifest,project/'outputs')
        save_json(cleanup_marker,{'deleted_targets':len(manifest['targets']),'smokes':list(smoke_roots)})
    os.environ.pop('SWANLAB_MODE',None)
    groundings={}
    for dataset in DATASETS:
        wait_for_gpus(args.poll_seconds)
        source=yaml.safe_load((project/'configs'/dataset/'grounding.yaml').read_text())
        destination=project/source['training']['output_dir']/stamp
        recipe=prepare_recipe(project,dataset,destination,batch_sizes[dataset])
        os.environ['SWANLAB_LOG_DIR']=str(destination/'swanlog')
        command=[*torchrun,'-m','hyprorec.scripts.grounding','--config',str(recipe)]
        run_stage(dataset+'-grounding',command,project,state_path,args.poll_seconds,retry_failed=True,log_directory=destination)
        check_grounding(destination,feature_paths[dataset])
        groundings[dataset]=str(destination.relative_to(project))
    update_configs(project,version,groundings)
    for dataset in DATASETS:
        for experiment in ['0p1','0p3','1p0','3p0']:
            name=dataset+'-lambda-'+experiment
            state=json.loads(state_path.read_text())
            trial_outputs=state.setdefault('trial_outputs',{})
            if name not in trial_outputs:
                recipe=yaml.safe_load((project/'configs'/dataset/('full-lambda-'+experiment+'.yaml')).read_text())
                base=Path(recipe['training']['output_dir'])
                if base.parts[:2]!=('outputs',dataset) or len(base.parts)!=3:
                    raise ValueError('Editable experiment output must be outputs/dataset/experiment')
                trial_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                trial_outputs[name]=str(base/trial_id)
                save_json(state_path,state)
            destination=project/trial_outputs[name]
            destination.mkdir(parents=True,exist_ok=True)
            config=destination/'run_config.yaml'
            if not config.exists():
                recipe=yaml.safe_load((project/'configs'/dataset/('full-lambda-'+experiment+'.yaml')).read_text())
                recipe['training']['output_dir']=str(destination.relative_to(project))
                recipe['training']['logging_dir']=str((destination/'trainer-logs').relative_to(project))
                config.write_text(yaml.safe_dump(recipe,sort_keys=False))
            wait_for_gpus(args.poll_seconds)
            validate_version(feature_paths[dataset],data_root/DATASETS[dataset])
            os.environ['SWANLAB_LOG_DIR']=str(destination/'swanlog')
            command=[*torchrun,'-m','hyprorec.scripts.train','--config',str(config)]
            run_stage(name,command,project,state_path,args.poll_seconds,retry_failed=True,log_directory=destination)
            if not all((destination/file).is_file() for file in ['eval_results.json','test_results.json']):
                raise ValueError(f'Final eval/test artifacts missing for {name}')
            (destination/'exit-code.txt').write_text('0\n')
    state=json.loads(state_path.read_text());state['complete']=True;save_json(state_path,state)
    print('Feature refresh, Grounding and both lambda queues completed',flush=True)


if __name__=='__main__':main()
