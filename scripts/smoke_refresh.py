"""Eight-rank Grounding training/save/restore and graph-text CRS smoke."""
from __future__ import annotations
import argparse
import copy
import gc
import faulthandler
import signal
import json
import math
import os
from pathlib import Path
import torch
from torch.utils.data import Subset
from transformers import AutoConfig, AutoTokenizer, set_seed
from hyprorec.config import parse_experiment_args
from hyprorec.configuration_grounding import HoCRSGroundingConfig
from hyprorec.data.grounding import GroundingSourceDataset, HoCRSGroundingDataset, HoCRSGroundingCollator
from hyprorec.data.hypergraph import HypergraphTable
from hyprorec.modeling_grounding import HoCRSGroundingModel, build_source_config
from hyprorec.scripts.grounding import GroundingTrainer, _load_feature_tables, _finalize_grounding
from hyprorec.model_factory import build_model
from hyprorec.backbones import load_backbone_adapter
from hyprorec.processing_hocrs import HoCRSProcessor
from hyprorec.data import HoCRSDataset, HoCRSDataCollator
from hyprorec.scripts.train import _dataset_config
from hyprorec.training import HyProRecTrainer


def free_trainer(trainer):
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.free_memory()


def grounding(model_args, data_args, training_args, config_path, output, tables, table):
    dimensions={view:values.size(1) for view,values in tables.items()}
    config=HoCRSGroundingConfig(input_dims=dimensions,
        hidden_dim=model_args.hypergraph_hidden_dim, output_dim=model_args.hypergraph_output_dim,
        num_layers=model_args.hypergraph_num_layers, dropout=model_args.hypergraph_dropout,
        temperature=model_args.grounding_temperature, lambda_ga_sa=model_args.grounding_ga_sa_weight,
        lambda_ga_sn=model_args.grounding_ga_sn_weight, lambda_sa_sn=model_args.grounding_sa_sn_weight)
    combined, selections, results = {}, {}, {}
    ids=torch.randperm(table.num_items, generator=torch.Generator().manual_seed(42))[:80].tolist()
    for view in config.views:
        set_seed(42)
        one=copy.deepcopy(config);one.views=(view,)
        tokenizer=None
        maximum=64
        if view=='txt':
            tokenizer=AutoTokenizer.from_pretrained(model_args.grounding_tokenizer_name_or_path)
            source_config=build_source_config(view,AutoConfig.from_pretrained(model_args.grounding_tokenizer_name_or_path))
            source_config.vocab_size=len(tokenizer)
            maximum=source_config.max_position_embeddings
            if source_config.model_type=='mpnet':maximum-=tokenizer.pad_token_id+1
            one.source_configs[view]=source_config.to_dict()
        source=GroundingSourceDataset(data_args.dataset_path,(view,))
        model=HoCRSGroundingModel(one)
        config.source_configs.update(one.source_configs)
        args=copy.deepcopy(training_args)
        args.output_dir=str(output/view)
        args.logging_dir=str(output/view/'trainer-logs')
        args.num_train_epochs=2
        args.max_steps=-1
        args.per_device_eval_batch_size=args.per_device_train_batch_size
        args.gradient_accumulation_steps=1
        args.logging_steps=1
        args.disable_tqdm=True
        args.report_to=[]
        args.dataloader_num_workers=0
        args.dataloader_pin_memory=False
        args.dataloader_prefetch_factor=None
        args.dataloader_persistent_workers=False
        args.eval_strategy='epoch';args.save_strategy='epoch'
        args.load_best_model_at_end=True
        args.ddp_find_unused_parameters=True
        trainer=GroundingTrainer(model=model,args=args,
            train_dataset=HoCRSGroundingDataset(table,(view,),data_args.topk,data_args.khop,ids[:64]),
            eval_dataset=HoCRSGroundingDataset(table,(view,),data_args.topk,data_args.khop,ids[64:]),
            data_collator=HoCRSGroundingCollator({view:tables[view]},(view,),source,tokenizer,maximum),
            grounding_view=view)
        print("GROUNDING TRAIN START",view,os.environ["RANK"],flush=True)
        faulthandler.dump_traceback_later(60,file=TRACE)
        trained=trainer.train()
        faulthandler.cancel_dump_traceback_later()
        print("GROUNDING TRAIN END",view,os.environ["RANK"],flush=True)
        evaluated=trainer.evaluate()
        if not math.isfinite(trained.metrics['train_loss']) or not math.isfinite(evaluated['eval_loss']):
            raise ValueError('Grounding loss must remain finite')
        trainer.save_model();trainer.save_state()
        if trainer.is_world_process_zero():
            combined.update({key:value.detach().cpu().clone() for key,value in model.state_dict().items()})
            selections[view]={'checkpoint':trainer.state.best_model_checkpoint,'eval_loss':trainer.state.best_metric}
            results[view]={'train_loss':trained.metrics['train_loss'],'eval_loss':evaluated['eval_loss'],
                           'steps':trainer.state.global_step,
                           'peak_memory_bytes':torch.cuda.max_memory_allocated()}
        free_trainer(trainer)
        del trainer,model,source
        gc.collect();torch.cuda.empty_cache()
    main=int(os.environ['RANK'])==0
    _finalize_grounding(main,str(output),config,combined,selections,config_path,model_args,data_args,training_args)
    torch.distributed.barrier()
    # A strict restore catches missing tower keys and mixed feature dimension errors.
    restored=HoCRSGroundingModel.from_pretrained(output)
    if restored.config.input_dims!=dimensions:
        raise ValueError('Restored Grounding dimensions changed')
    del restored,combined
    gc.collect()
    return results


def crs(project,dataset,output,checkpoint,tables,table):
    model_args,data_args,args,config_path=parse_experiment_args(['--config',str(project/'configs'/dataset/'full-lambda-0p1.yaml')])
    model_args.grounding_checkpoint_path=str(checkpoint)
    adapter=load_backbone_adapter(model_args.backbone_name_or_path)
    processor=HoCRSProcessor(adapter.tokenizer,model_args.num_prompt_tokens)
    model=build_model(model_args,data_args,processor,tables,adapter)
    data=HoCRSDataset(_dataset_config(data_args,data_args.hyperedge_table_path,training=True),'train',table)
    collator=HoCRSDataCollator(processor,data_args.max_history_tokens,data_args.max_sequence_tokens,views=tuple(data_args.views))
    args.output_dir=str(output)
    args.logging_dir=str(output/'trainer-logs')
    args.max_steps=2
    args.gradient_accumulation_steps=1
    args.eval_strategy='no';args.save_strategy='no'
    args.load_best_model_at_end=False
    args.dataloader_num_workers=0
    args.dataloader_persistent_workers=False
    args.dataloader_prefetch_factor=None
    args.prediction_loss_only=True
    args.logging_steps=1;args.disable_tqdm=True;args.report_to=[]
    args.label_names=['labels','rec_labels']
    trainer=HyProRecTrainer(model=model,args=args,processing_class=processor,
        train_dataset=Subset(data,list(range(32))),eval_dataset=Subset(data,list(range(16))),
        data_collator=collator,attention_token_ids=processor.get_token_id_map())
    trained=trainer.train()
    evaluated=trainer.evaluate()
    if not math.isfinite(trained.metrics['train_loss']) or not math.isfinite(evaluated['eval_loss']):
        raise ValueError('CRS losses must remain finite')
    batch=collator([data[0]]).to(args.device)
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        encoded=model.encode(batch['input_ids'],attention_mask=batch['attention_mask'],
                             hypergraphs=batch['hypergraphs'],use_cache=True)
        generated=model.generate(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                                 hypergraphs=batch['hypergraphs'],max_new_tokens=2,use_cache=True)
    if encoded.past_key_values is None or generated.size(1)<batch['input_ids'].size(1)+1:
        raise ValueError('Cached graph-text generation failed')
    result={'train_loss':trained.metrics['train_loss'],'eval_loss':evaluated['eval_loss'],
            'cached_generation':True,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
    free_trainer(trainer)
    del trainer,model,adapter,encoded,generated,batch
    gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['redial','tg_redial'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--fixture',action='store_true',help='Explicit mixed-dimension interface preflight before feature publication')
    cli=parser.parse_args()
    global TRACE
    cli.output.mkdir(parents=True,exist_ok=True)
    TRACE=(cli.output/("rank-"+os.environ["RANK"]+".trace.log")).open("w")
    faulthandler.register(signal.SIGUSR1,file=TRACE,all_threads=True)
    project=Path(__file__).resolve().parents[1]
    os.environ['SWANLAB_MODE']='disabled'
    torch.set_num_threads(1)
    model_args,data_args,args,config_path=parse_experiment_args(['--config',str(project/'configs'/cli.dataset/'grounding.yaml')])
    tables=_load_feature_tables(data_args)
    if cli.fixture:
        for view in ['ado','vdo']:tables[view]=torch.nn.functional.normalize(tables[view][:,:512],dim=-1)
    table=HypergraphTable.from_json(data_args.hyperedge_table_path)
    # TrainingArguments initializes the eight-rank process group.
    device=args.device
    if torch.distributed.get_world_size()!=8:
        raise ValueError('Smoke requires all eight DDP ranks')
    cli.output.mkdir(parents=True,exist_ok=True)
    args.per_device_train_batch_size=cli.batch_size
    torch.cuda.reset_peak_memory_stats()
    losses=grounding(model_args,data_args,args,config_path,cli.output/'grounding',tables,table)
    torch.distributed.barrier()
    stats=crs(project,cli.dataset,cli.output/'crs',cli.output/'grounding',tables,table)
    ranks=[None]*8
    torch.distributed.all_gather_object(ranks,{'rank':torch.distributed.get_rank(),'passed':True})
    if torch.distributed.get_rank()==0:
        (cli.output/'smoke.json').write_text(json.dumps({'passed':True,'world_size':8,
            'fixture':cli.fixture,'ranks':ranks,'grounding':losses,'crs':stats},indent=2))
    torch.distributed.barrier()
    print('SMOKE finished rank',torch.distributed.get_rank(),flush=True)


if __name__=='__main__':main()
