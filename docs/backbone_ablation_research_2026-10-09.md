# HyProRec 小参数 backbone 消融调研

调研日期：2026-10-09。依据当前工作区代码、实验配置、模型发布方的 Hugging Face 模型卡与配置，以及仓库锁定的 Transformers v5.14.1 官方实现。本文是接入与实验设计建议；没有下载模型权重、运行 GPU 实验或测量候选模型显存。

## 最终主实验名单

结合用户明确接受 2×4 HSDP、并要求考虑 LLaVA 的条件，最终建议固定为 **9 个 backbone：4 个四模态/音视频理解模型、3 个 VLM、2 个文本 LLM**。下面是主实验名单；后文其他模型属于候选池与筛选记录。

| 官方 HF ID | 类型与规模 | 实验角色 | 起步资源策略 |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-Omni-3B` | Omni；3B 档，完整发布模型约 6B；当前代码只加载 Thinker | 已有主模型；同代 Qwen 三组对照的核心 | 沿用当前 DDP |
| `Qwen/Qwen2.5-VL-3B-Instruct` | 图文/视频 VLM；整模约 3.8B | 检验 Omni 与视觉语言训练的差异 | DDP 优先 |
| `Qwen/Qwen2.5-3B-Instruct` | 文本 LLM；3.09B | 必做的主要文本对照，最接近现有 3B 档 | DDP 优先 |
| `meta-llama/Llama-3.2-3B-Instruct` | 文本 LLM；3B 档 | 非 Qwen 文本对照，考察语言模型家族泛化 | DDP 优先 |
| `OpenGVLab/InternVL3_5-2B-HF` | 图文/视频 VLM；整模 2.3B | 主流 InternVL 家族及较低容量点 | DDP 优先 |
| `sensenova/InteractiveOmni-4B` | 四模态 Omni；4B 档，HF 整模统计约 6B | 增加面向音视频多轮对话的官方 Omni | 2×4 HSDP 为资源预案 |
| `tsinghua-ee/video-SALMONN-2-Pro-4B` | 音视频理解；Qwen3-VL 底座，HF 整模统计约 5B | 增加以音视频联合理解为重点的模型 | 2×4 HSDP 为资源预案 |
| `openbmb/MiniCPM-o-4_5` | 四模态 Omni；Qwen3-8B 底座，整模 9B | 主流 MiniCPM-o 家族；较大容量的扩展验证 | 优先 2×4 HSDP |
| `llava-hf/llava-1.5-7b-hf` | 图文 VLM；Vicuna v1.5 的 7B 档语言模型 + CLIP 视觉塔 | 经典 LLaVA 基线，增加非 Qwen 的多模态语言底座 | 优先 2×4 HSDP |

LLaVA 选择 [HF 集成版本 llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)，权重来源是 [原作者发布的 liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)。原团队 GitHub 明确使用该作者仓库；虽然账号形式是个人名称，它是原作者发布的模型，不是私人缩放衍生版。[官方项目](https://github.com/haotian-liu/LLaVA)。采用经典 1.5 的目的在于增加 Vicuna/LLaMA 系列的多模态基线与文献可比性；OneVision-Qwen2-7B 也可作为后续较新版本，但其 Qwen 底座会增加本名单的家族重合。

最重要的机制对照固定为 **Qwen2.5 文本 3B / VL-3B / Omni-3B**，相同组件、物品表、损失权重和 checkpoint 选择协议。其他模型用于泛化与规模扩展；尤其不能将 MiniCPM-o-4.5 相对 3B 模型的绝对增益全部归因于 Omni 训练。

如果还需一个与 4B 组更接近的同代文本对照，可追加 `Qwen/Qwen3-4B-Instruct-2507`，作为第 10 个模型。它是补充实验，不改变上述主名单。

本轮不将 TLive-Omni、Gemma 的 PLE 模型、Phi 的模态 LoRA 模型或 LLaVA 0.5B 放入主名单：分别考虑领域训练与混合注意力接入、soft token 接口适配、模态路由变量、以及与主模型过大的容量差。相关细节与官方来源保留在后文。

资源策略是工程预案，尚未测得候选峰值。最终运行必须确认冻结 decoder 的输入梯度、模型逐层分片、保存重载与 tokenizer 扩表；官方 checkpoint 并不意味着当前 `load_backbone()` 可直接加载所有候选。若抽取语言部分，所有 MLLM 使用一致规则并保留其多模态训练后的 decoder 与原 LM head；论文分别报告发布参数量、实际驻留参数量和外挂可训练参数量。

## 算力约束更新：接受 2×4 HSDP

用户已明确接受 8 张 RTX 3090 使用 2×4 HSDP。因此，原先以单卡 DDP 为依据的小于 7B 筛选边界应放宽：小模型仍优先，但 7–9B 语言底座、约 8–12B 整模的主流官方 Omni 也可以进入候选。

这里定义 2×4 为 **2 个复制组、每组 4 卡分片**，即 `dp_replicate_size=2`、`dp_shard_size=4`。当前 `src/hyprorec/arguments.py` 已有这两个字段；分片数大于 1 时启用 FSDP2 与 ParallelismConfig。但读取配置代码不等于已经验证真实 8 卡训练、自动包裹、保存与恢复均能跑通。

按照均一 BF16 权重、所有相关参数被四路分片估算，每卡常驻权重下限约为：

| 实际驻留参数 | 四路分片后的权重下限 |
| --- | --- |
| 8B | 3.7 GiB |
| 9B | 4.2 GiB |
| 12B | 5.6 GiB |

该表不含逐层 all-gather 后临时完整权重、激活、预取缓冲、外挂组件优化器状态或未分片的特征表。FSDP/HSDP 分片不会自动将每卡激活除以四。对本项目的冻结 backbone，主要收益来自权重分片，反向传播仍须经过 backbone 更新图 soft tokens。

按 decoder block 包裹是资源判断的重要条件；如果把整套 backbone 包成一个单元，all-gather 时可能重新出现整模显存峰值。单机 8 卡选择两组四卡还应依据 GPU/PCIe 拓扑，不能预设 2×4 比 1×8 FSDP 更快。HSDP 中各分片 rank 仍处理不同数据，所以 micro batch=1、累积 16、8 个纯数据并行 rank 的有效 batch 仍为 128。

在这一新约束下，将 **MiniCPM-o-4.5（官方整模 9B）** 提升为优先的额外 Omni；Qwen2.5-Omni-7B、InternOmni 和 Baichuan-Omni-1.5 也恢复为可考虑候选，具体取决于实际保留模块与测得的峰值/吞吐。对 Qwen2.5-Omni-7B，要区分 7B 档语言模型与约 11B 的完整发布模型；当前仓库仅加载 Thinker。30B-A3B 等大 MoE 仍放在后续，原因是总驻留权重与通信成本，不是激活参数口径。

依据：[PyTorch 2.6 fully_shard 的二维 mesh / HSDP 说明](https://docs.pytorch.org/docs/2.6/distributed.fsdp.fully_shard.html)、[HF Accelerate FSDP 文档](https://huggingface.co/docs/accelerate/usage_guides/fsdp)。本文后续“暂缓大于 7B”的条目保留了初轮筛选理由，应以本节更新后的资源约束重新评估。

## 0. 补充核查：四模态 Omni 优先候选

针对“是否还有文本、图像、音频、视频四模态开源 Omni”的追问，补充核查发现初轮筛选遗漏了 **InteractiveOmni-4B、TLive-Omni-4B 和 video-SALMONN-2-Pro-4B**。它们均有发布团队的官方 HF 权重。若优先四模态覆盖，应把这些模型加入主候选，而不只采用初轮 VLM 组。

| 官方模型 | 四模态能力与规模 | 接入优先级 |
| --- | --- | --- |
| [sensenova/InteractiveOmni-4B](https://huggingface.co/sensenova/InteractiveOmni-4B) | 商汤官方；明确支持文本、图像、音频、视频联合输入与文本/语音输出；HF 权重统计约 6B，而非整模 4B | 值得优先新增；custom code，官方要求 FA2；核对语言解码器与语音生成部分的拆分 |
| [TaoLiveAIGC/TLive-Omni-4B](https://huggingface.co/TaoLiveAIGC/TLive-Omni-4B) | 原团队官方；基于 Qwen3.5，支持四模态及视频音轨联合理解；HF 整模统计约 6B；面向电商直播理解 | 值得新增；领域训练是需要披露的变量，另需处理混合注意力内核；官方 vLLM wheel 面向 H20，不能直接照搬到 3090 |
| [tsinghua-ee/video-SALMONN-2-Pro-4B](https://huggingface.co/tsinghua-ee/video-SALMONN-2-Pro-4B) | 清华原团队官方；Qwen3-VL 加音频分支，保留图文能力并训练音视频联合理解；HF 整模统计约 5B | 值得优先新增；音视频理解型，文本输出，不要求具备语音生成能力 |
| [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | Google 官方；图像、视频帧、音频、文本输入，文本输出；官方含 embedding 参数为 5.1B | 官方主流家族；需要显式适配 PLE，不能直接套用任意 inputs_embeds |
| [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) | Google 官方；文本、图像、视频、音频输入；raw 参数口径约 6B，E2B 不是总驻留参数 | 可作为补充；同样需要核对逐层 embedding 与 offload 路径 |
| [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | Microsoft 官方；文本、图像、音频输入，视频通过多帧理解；约 5.5–5.6B，受保留的 adapter 影响 | 主流且有参考价值，但要区分多帧视频理解与带音轨视频的时间对齐，并处理模态 LoRA 路由 |

加上现有 Qwen2.5-Omni-3B，已经有多个小型 Omni 候选。官方参数统计仍不是训练峰值；完整 5–6B 权重加输入梯度激活可能使 24GB 较紧张。按现有纯图 token 输入路径保留训练后的语言 decoder 与原 LM head，可以降低不参与前向的感知/生成模块的权重开销。

若要一个四模态优先的 5 模型组，可选择 Qwen2.5-Omni-3B、InteractiveOmni-4B、TLive-Omni-4B、video-SALMONN-2-Pro-4B、Gemma-4-E2B，再保留 Qwen2.5-3B-Instruct 文本对照。这是 5 个全模态 checkpoint，其中若干沿用 Qwen 语言底座，不代表 5 个独立语言模型家族。官方权重、HF 原生 Transformers 集成、广泛生态使用是三个不同条件；新增三个项目满足原团队官方权重条件，但接入成熟度仍需核对。

其他核查结果：

- [video-SALMONN-2_plus_3B](https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_3B) 也支持音视频理解，但 HF 仓库以 adapter 形式发布，基础音频对齐模型需要按官方流程准备。应优先完整发布的 Pro-4B，以减少重建成本。
- [Ming-Lite-Omni](https://huggingface.co/inclusionAI/Ming-Lite-Omni) 的 2.8B 是激活参数。官方记录 BF16 加载约 40890MB，超出单张 3090 的显存，不能按 dense 3B 排队。
- [MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5) 官方总计 9B；[InternOmni](https://huggingface.co/OpenGVLab/InternOmni) HF 统计约 9B；[Baichuan-Omni-1d5](https://huggingface.co/baichuan-inc/Baichuan-Omni-1d5) 已是 7B 档并带多模态模块。本轮小规模优先组暂缓。
- [Megrez-3B-Omni](https://huggingface.co/Infinigence/Megrez-3B-Omni) 的官方说明明确覆盖文本、图像、音频三种模态，没有充分依据将它当成独立视频能力已确认的四模态主模型。
- [Ovis-Omni-Embedding-3B](https://huggingface.co/ATH-MaaS/Ovis-Omni-Embedding-3B) 支持四模态，但属于检索 embedding 模型，移除了 LM head；不适合直接替换当前推荐与对话联合损失的生成 backbone。

## 1. 初轮兼顾 VLM 与文本 LLM 的候选

建议准备 **6 个 MLLM + 2 个文本 LLM**，总计 8 个 backbone。先用除 MiniCPM-V-4 之外的 7 个完成主体实验，随后用 MiniCPM-V-4 补充一个不同语言模型家族。所有候选均为模型发布团队的 checkpoint；不使用个人缩放、蒸馏或量化衍生版本。

| 模型及官方 HF 仓库 | 参数口径 | 原生输入能力 | 对本项目的价值 | 接入与资源判断 |
| --- | --- | --- | --- | --- |
| [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | 3B 档语言模型；Thinker 还包含视觉、音频塔 | 文本、图像、视频、音频 | 已有主模型与参考结果 | 已接入 Thinker；用户报告约 80% 显存 |
| [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 语言模型约 3.09B；整模约 3.8B | 文本、图像、视频 | 与现有 Omni、文本 Qwen2.5 组成同代、相近容量的主要对照 | HF 原生；增加 ConditionalGeneration 加载分支；优先 |
| [OpenGVLab/InternVL3_5-2B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF) | 官方表：语言 2.0B、视觉 0.3B、总计 2.3B | 文本、图像、视频 | 主流视觉语言模型家族；较低资源压力 | 优先官方 `-HF` 格式，使用 InternVLForConditionalGeneration；优先 |
| [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | 2.2B 档整模；语言部分为 SmolLM2-1.7B 系列 | 文本、图像、多图、视频 | HF 官方小型 VLM；不同于 Qwen 的语言模型来源 | HF 原生；必须显式指定 BF16，官方整模配置为 FP32；优先 |
| [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | 4B 档；预算时计入视觉塔，最终以加载后统计为准 | 文本、图像 | 增加 Google 家族，避免候选过度集中于 Qwen 系列 | HF 原生 Gemma3ForConditionalGeneration；中等资源压力；需已有模型访问权限 |
| [openbmb/MiniCPM-V-4](https://huggingface.co/openbmb/MiniCPM-V-4) | MiniCPM4-3B 系列语言模型 + SigLIP2-400M，官方总计 4.1B | 文本、图像、多图、视频 | 增加 MiniCPM 家族；整体规模仍低于 7B | 官方 custom code；外层会丢弃 inputs_embeds，接入经过多模态训练的 `.llm`；第二批 |
| [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 官方 3.09B | 文本 | **必须加入的主要文本对照** | AutoModelForCausalLM；当前加载路径可用，适配成本最低 |
| [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 3B 档 | 文本 | 第二个相近参数量文本对照，考察跨家族泛化 | AutoModelForCausalLM；需已有模型访问权限 |

这里的显存判断是根据参数量、架构和现有实验条件作出的筛选，不能当作跑通保证。起步统一 micro batch=1、BF16、非 reentrant gradient checkpointing、冻结 backbone、关闭训练缓存、长度上限 1024。实际峰值要包含最长样本、第一次 optimizer step 和 evaluation。

Qwen2.5-VL 的整模约 3.8B，也见 [MiniCPM 官方模型卡的参数比较表](https://huggingface.co/openbmb/MiniCPM-V-4)。InternVL 官方表明确区分语言与视觉参数，不能仅根据模型名中的 2B 或 4B 计算显存。

## 2. 当前方法实际使用 backbone 的方式

代码入口：`src/hyprorec/scripts/train.py`、`model_factory.py`、`modeling_hocrs.py`、`processing_hocrs.py`、`data/redial.py`、`modeling_grounding.py`。

### 数据与图表示

`prepare_embeddings.py` 默认用四组独立的预训练编码器产生离线物品特征：MPNet 文本、ViT 图像、Wav2Vec2 音频、VideoMAE 视频。它们与 CRS backbone 的原生感知塔不是同一条路径。

Grounding 阶段只训练 `txt`、`img`、`ado`、`vdo` 四个独立语义图塔。图锚点、源模态锚点、源邻居聚合之间使用多正样本对比损失；源编码器通过配置构造并参与 Grounding 训练。CRS 阶段加载 Grounding checkpoint 后，`load_grounding_checkpoint()` 会冻结所加载的超图 encoder。

CRS 输入组织为：

```text
20 个可训练 soft prompt
  → 截断后的对话历史
  → 各视图的节点与超边 token
  → 回答角色标记
  → teacher-forced 回答
```

节点和超边先由图 encoder 编码，再分别投影到 backbone hidden size，通过 GraphTokenMoE 的残差专家变换，替换占位 token 的 embedding。`encode()` 把得到的 `inputs_embeds` 送入 backbone。

当前 batch 不含 backbone 所需的原始 `pixel_values`、视频帧或音频特征。因此，虽然 backbone checkpoint 是 Omni/VLM，实际 CRS 前向主要使用其经过多模态训练的语言解码器。MLLM 原生编码器可能仍然占据权重显存，但其原生感知能力没有被这条训练输入路径直接调用。

### 推荐与对话目标

推荐头在最后一层 hidden states 上做各视图的可学习 attention pooling，再将用户表示与投影、归一化后的物品特征表打分。一个样本级 router 融合各视图分数。

推荐 pooling mask 排除了 teacher-forced 回答，推荐表示来自 prompt 与历史，避免读取目标回答。对话 labels 则忽略 prompt，只监督回答 token。

默认主要目标实际为：

\[
L = CE(s_{fuse},y)
  + \eta\frac{1}{|\mathcal M|}\sum_v CE(s_v,y)
  + \lambda L_{conv}.
\]

代码对 view CE 使用 **平均**，不是文稿公式中的简单求和。单视图时额外 view loss 为零。BPR 与 router balance 在主配置中关闭。

CRS 可训练组件主要包括 soft prompt、图 projector、GraphTokenMoE、attention pooler、用户/物品投影及推荐 router。backbone 和加载 Grounding 权重的图 encoder 被冻结。

冻结 backbone 仍需要保留输入到输出的梯度：对话和推荐损失都要穿过解码器更新 soft tokens。把整个 backbone 放进 `no_grad()` 或 `inference_mode()` 会切断这条训练路径。

### 可以如何解释消融结果

这套实验直接检验的是：**多模态训练过的语言解码器，是否更适合接收外部多模态超图 soft tokens**。

若文本 LLM 也获得稳定收益，可以支持 HyProRec 作为多模态图表示到冻结文本模型的适配组件。结论范围应限定在本任务与本表示路径；当前实验本身不能证明原始图片、音频或视频的一般感知能力。

## 3. 与 8×RTX 3090 的关系

### 当前 3B / 7B 配置

两者共同使用：冻结 backbone、micro batch=1、八卡正式训练时梯度累积 32（有效 update batch=256）、BF16、gradient checkpointing、`use_reentrant: false`、`ddp_find_unused_parameters: true`、历史上限 256、序列上限 1024、每个局部图最大节点数 64。`encode()` 固定 `use_cache=False`。

7B 配置没有额外指定 FSDP/HSDP、量化、CPU offload 或 FlashAttention。已有 checkpointing 逻辑会在训练时让冻结 decoder 保持 checkpoint 所需的 training 状态，并关闭其中的 Dropout。

有两处与 3B 不同、会影响实验结论的设置：

| 设置 | 3B 主配置 | 7B 配置 |
| --- | --- | --- |
| conversation_loss_weight | 1.0 | 0.1 |
| 最佳 checkpoint 指标 | eval_loss，越低越好 | eval_recall@50，越高越好 |

因此现有两组不能直接作为只替换 backbone 的严格对照。`train.py` 还会动态将 eval/save 周期设为每个 epoch 大约四次，覆盖 YAML 中的固定步数。

### DDP 的限制与预算

当前 torchrun 的 8 个进程走 DDP：每张卡各放一份 backbone。8×24GB 不会自动合并成一个可用的 192GB 显存池。micro batch=1、累积 16、8 卡时有效 batch 为 128。

冻结 backbone 通常不需要其参数梯度和 Adam 状态，但仍需要权重、用于输入梯度的激活、重计算临时空间、外挂组件梯度和优化器状态、词表 logits、CUDA workspace 等。

均一 BF16 权重的粗略下限为：

\[
M_{weights}\approx\frac{2P}{2^{30}}\ \mathrm{GiB},
\]

即 3B 约 5.6GiB、4B 约 7.5GiB、7B 约 13.0GiB。该下限不包含多模态塔的额外参数或训练开销。不能把当前 3B 的约 80% 显存按参数比例线性外推。

最适合优先试的是 **1.7–4B 左右的语言解码器、整模约 2–5B 的 dense checkpoint**。7B 是当前工程条件下的保守边界，不是 3090 上冻结模型训练的理论上限。

### 接入时优先处理的显存项

1. 对所有候选显式指定 backbone 权重 `dtype=torch.bfloat16`。仓库的 Transformers v5.14.1 在未传 dtype 时默认走 auto；SmolVLM2 的顶层配置是 float32，auto 会造成额外权重显存。训练参数 `bf16: true` 不能替代加载权重时的 dtype 设置。[官方 v5.14.1 加载实现](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/modeling_utils.py)、[SmolVLM2 配置](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct/blob/main/config.json)。
2. 核对实际 attention backend。当前依赖没有 flash-attn，也未显式选择 `attn_implementation`；不能假定正在用 FA2。先使用各模型支持的 SDPA，必要时再处理 FA2 的依赖与兼容性。
3. 对纯图 token 路径，可以接官方多模态 checkpoint 中的语言解码器与原有 LM head，省去不调用的视觉/音频塔。Omni 目前已只加载 Thinker，没有加载 Talker/Token2Wav；但 Thinker 本身仍有 `audio_tower` 和 `visual`。[官方 Thinker 实现](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py)。
4. 如果采用只保留解码器的方式，所有 MLLM 应统一此规则，记录来源 checkpoint、保留参数量及保存格式。保留多模态训练后的权重，不重新初始化为对应文本基础模型。
5. `output_hidden_states=True` 请求所有层，而推荐头只用最后一层；forward 后再把 tuple 缩为最后一层，无法消除前向期间的峰值。可在后续接入时只获取末层，但它只是显存预算的一部分。
6. 词表大小也影响 logits 与 CE 临时空间。训练只在最终返回时隐藏 conversation_logits，并不表示 LM 前向没有创建词表 logits。

若上述小模型仍超出 24GB，再分析具体峰值来源：权重占主导时考虑 FSDP/HSDP 分片；激活占主导时先处理序列、checkpointing 与 attention。推理示例中的 `device_map="auto"` 不能直接作为当前 DDP 训练的替代配置。

## 4. 文本对照与消融设计

### 必做的 Qwen2.5 对照

| Backbone | 文本图 tokens | 四模态图 tokens |
| --- | --- | --- |
| Qwen2.5-3B-Instruct | A | B |
| Qwen2.5-VL-3B-Instruct | C | D |
| Qwen2.5-Omni-3B Thinker | E | F |

以固定的 recommendation head、物品表和超参数，比较 B−A、D−C、F−E 的多模态图增益，再比较不同 backbone 的增益。这样比只看 F 与 B 的绝对 Recall 更接近“MLLM 是否更适配外挂组件”这一问题。

这是同代、容量接近的比较，不是完全相同参数与训练数据的因果实验。Omni 配置的文本 embedding 与 LM head 不共享权重，而文本 Qwen2.5-3B 共享；Omni 文本部分按结构估计约 3.4B，文本对照官方是 3.09B。需在论文表格披露 embedding tying、decoder 参数与总驻留参数。[Omni 官方配置](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/main/config.json)、[Qwen2.5-3B 官方参数说明](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。

### 物品表是另一条多模态路径

即使图 `views=[txt]`，若 `item_table_view=full`，推荐头仍然使用图像、音频、视频物品特征。这只能称为“文本图输入”，不能称为整个系统的纯文本实验。

当前 `only-txt-item-txt-attn.yaml` 同时改变图与物品表。建议先在图输入对照中固定物品表；若要定位增益来自哪里，再对主模型及主要文本对照补做：

| 图输入 | 物品表 | 解释 |
| --- | --- | --- |
| txt | txt | 文本语义配置 |
| txt/img/ado/vdo | txt | 考察多模态图经过 backbone 的贡献 |
| txt | full | 考察推荐头多模态物品表的贡献 |
| txt/img/ado/vdo | full | 完整系统 |

图视图数量改变也会改变 token 数与信息容量。若需要进一步排除“只是更多 token”的解释，可增加保持图位置和长度、打乱非文本物品特征对应关系的控制。推荐头、view loss 定义和可训练参数变化都应随实验报告。

### 跨家族与成对补充

Llama-3.2-3B-Instruct 用于跨家族文本验证。已有 DialoGPT-small 可以保留为历史小模型参考，但它的容量与预训练年代不足以单独承担 MLLM 对 LLM 的主要结论。

预算允许时，补做 [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) 与 SmolVLM2：语言部分来源更接近。也可以用 [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 对照 Qwen3-VL-4B-Instruct；两者仍有训练配方、位置编码等差异。

### 统一实验条件

- 相同数据划分、离线特征、Grounding checkpoint、物品 ID 顺序、图采样与种子。
- 相同 soft prompt 数、图节点预算、训练轮数、有效 batch、LR 搜索预算和冻结策略。
- 统一 lambda；推荐以同一小范围验证集选择协议筛选，避免为各模型使用不等的调参预算。已有 3B=1.0 与 7B=0.1 应重做成同协议对照。
- 推荐任务统一用 validation Recall@50 选择 checkpoint，再在 test 上报告 Recall/MRR/NDCG。不同 tokenizer 的 CE 数值不宜用作跨模型优劣指标。
- 当前 prompt 是自定义角色文本，没有调用 backbone 的官方 chat template。先统一现有格式做主消融；如使用各家模板，需统一构造语义、图插入位置和回答 mask，并说明协议。
- 同一个 `max_history_tokens=256` 在不同 tokenizer 下会保留不同内容；记录实际保留的轮次/文本范围与 token 分位数。`max_sequence_tokens=1024` 是超长检查而非自动截断；不应静默丢弃某个 tokenizer 下的超长样本。
- 外挂模块的尺寸跟随 backbone hidden size，可训练参数量不会完全相同。统一中间维度并报告外挂参数，必要时再补容量匹配对照。
- 当前对话 BLEU/Distinct 是 teacher-forcing 预测指标；如果论证实际对话生成能力，需要独立的自回归生成评估。
- 正式结果建议至少 3 个随机种子；记录每卡峰值 allocated/reserved 显存、samples/s、训练时长和有效样本数。

## 5. 额外候选与暂缓项

| 模型 | 已核实的规模/特点 | 建议 |
| --- | --- | --- |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) / [4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 官方 dense 小模型；2B 配置 H=2048、28 层；4B H=2560、36 层 | 想加入更新一代主流 VLM 时优先，2B 更适合资源筛选；4B 可配文本 Qwen3-4B |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) / [4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 原生视觉语言模型，不是纯文本对照；dense 小模型也使用 Gated DeltaNet 混合注意力 | 可作为较新补充，但先核对训练内核；缺少 causal_conv1d/fla 会走更慢、更耗显存的参考路径 |
| [InternVL3_5-4B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-4B-HF) | 语言 4.4B、视觉 0.3B、总计 4.7B | 若 2B 接入顺利且资源有余量，可补同家族规模消融 |
| [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | 3.8B 语言底座；加视觉/音频模块与模态 LoRA 后约 5.5–5.6B，随加载的 adapter 变化 | 官方、较小、支持图像/音频；但需要明确 LoRA 路由。纯文本/图 token 路径未必激活多模态 adapter，解释复杂，第二批 |
| [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) | E2B 是有效参数口径；模型卡 raw 参数约 6B，HF 页面统计约 5B；支持视觉和音频 | 不按 2B 预算。其低内存部署依赖 offload 等专门路径，逐层 embedding 也增加 soft token 接入复杂度，暂缓 |
| [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | 官方 2.3B effective、含 embedding 为 5.1B；原生图像/视频/音频输入 | 小型新 Omni 候选，但 PLE 不兼容直接替换任意 inputs_embeds，需专门接入；暂缓 |
| [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) | 基于 Qwen3.5-0.8B 与 SigLIP2-400M；属于约 1B 档轻量模型 | 官方最新轻量补充，但既有混合注意力依赖，又与 3B 容量差较大；作为低容量参考 |
| [lmms-lab/llava-onevision-qwen2-0.5b-ov](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov) | 原团队小型 OneVision；语言约 0.5B，加 SO400M 视觉塔后约 0.9B | 满足主流 LLaVA 家族与小参数要求，但容量远低于主对照；作为历史/低容量参考 |

LLaVA 原团队版本可对应使用 [llava-hf 的 Transformers 格式转换](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)。它是 HF 集成格式转换，应明确来源是 lmms-lab；不是私人缩放模型。经典 LLaVA-1.5/NeXT 7B 加视觉塔后也需额外预算，不适合优先于上述 2–4B 候选。

Qwen3.5 的训练内核依据：[官方 Transformers 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)。Gemma-4 的 PLE 依据：[官方 v5.14.1 实现](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gemma4/modeling_gemma4.py)：只传 inputs_embeds 时会尝试匹配原词嵌入以恢复 token ID，自定义图 embedding 无法满足这种匹配；应显式处理 `per_layer_inputs`，而不是照搬 Omni 的同时传 input_ids 和 inputs_embeds。

Phi 参数近似来自 [官方技术报告](https://arxiv.org/html/2503.01743v2) 的 3.8B 底座、约 440M 视觉模块、370M 视觉 LoRA、460M 音频模块、460M 音频 LoRA 相加。其多模态适配主要依赖 mixture of LoRAs，因此抽取底座时需披露哪些 adapter 得到保留或激活。

[Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) 也已核实：语言 3.4B + 视觉 0.4B。但官方 Instruct 权重是 FP8，3090 不宜照搬 FP8 推理方案。官方提供 `FineGrainedFP8Config(dequantize=True)` 转 BF16 路径；该工程和数值差异使其适合后续补充，主对照先采用官方 BF16/FP32 原始 checkpoint。

不因名称中的 active 参数小而纳入大 MoE，例如 30B-A3B 仍需要存放远大于 3B 的权重。MiniCPM-o-2.6、MiniCPM-o-4.5、MiniCPM-V-4.5 等也不属于本轮优先的小规模组。

## 6. 接入改动边界

当前 refactor 已实现统一的 tokenizer/embedding/decoder hidden/LM logits 路径，并对 Qwen2.5-Omni Thinker 完成了实际加载与生成 smoke。后续接入其他模型仍需处理以下明确接口：

1. `load_backbone()` 与 `_backbone_from_config()` 同步支持模型类型，确保训练、保存、重载一致。
2. 统一取得 text config、input embedding、decoder hidden states 和原 LM head，保留训练过的权重。
3. 对不同模型核对 embedding scaling、position_ids、attention mask、special token 扩表与 checkpointing 行为。
4. 避免全局套用“同时传 input_ids 和 inputs_embeds”。当前 Omni 有特例；其他模型可能只允许二者之一。
5. MiniCPM-V-4 需调用 `.llm`；官方外层 forward 明确删除 inputs_embeds。[官方源码](https://huggingface.co/openbmb/MiniCPM-V-4/blob/main/modeling_minicpmv.py)。
6. Qwen VL、InternVL、SmolVLM、LLaVA 的 AutoModelForCausalLM 加载不能直接复用。Gemma3 则已在 v5.14.1 的该 Auto 映射中注册，仍需检查其嵌入缩放与前向接口。[官方 Auto 映射](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/auto/modeling_auto.py)。

建议实际执行顺序：Qwen2.5 文本 3B → Qwen2.5-VL-3B → InternVL3.5-2B-HF / SmolVLM2 → Llama3.2 文本 3B / Gemma3-4B → MiniCPM-V-4。先形成主要同代对照，再增加跨家族覆盖，最后考虑混合注意力、PLE 或模态 LoRA 的候选。
