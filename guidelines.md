# LAB操作规范：（可以创建一份与本项目无关的 local SKILL,专门用于指导登陆以及操作边际）

免密登陆： ssh lab
有tmux窗口，可以开启tmux窗口进行操作
/lhf/init.sh是很重要的脚本，有指定HF登陆、W&B登陆，以及HF HOME，GIT凭证。你可以阅读脚本内容，脚本并不是很完善，你可以灵活使用，但不要把我的凭证、API删除了
务必确保W&B、GIT提交账号为linexox（如果有后缀前缀，都可以，可能叫linexox7,也有可以能后面跟了学校名字，但是一定有linexox）
HOME盘已经爆满了，所以新环境，下载的文件都需要放在/data2/lhf中。这是实验室的服务器，所以不要超出lhf（/data2/lhf以及/lhf文件夹工作。

# 方法论一览

$\mathcal I=\{1,\ldots,N\}$ 为物品集合，$\mathcal M=(\mathrm{txt},\mathrm{img},\mathrm{ado},\mathrm{vdo})$ 为模态集合；$r_i^m,x_i^m$ 为原始模态输入与固定离线特征；$c,\mathcal I_c,y,w_{1:T}$ 为对话历史、历史物品、目标物品与目标回复。$\operatorname{norm}$ 表示 $\ell_2$ 归一化，$[\cdot;\cdot]$ 表示拼接。

## 1. Grounding

### 超图表示

$$
\begin{aligned}
x_i^m&=\operatorname{norm}(F_{\mathrm{off}}^m(r_i^m)),\\
\mathcal N_K^m(i)&=\operatorname{TopK}_{j\ne i}\cos(x_i^m,x_j^m),\\
e_i^m&=\{i\}\cup\mathcal N_K^m(i),\qquad B_{ve}^m=\mathbf1[v\in e].
\end{aligned}
$$

$D_V,D_E$ 为节点度与超边度矩阵。各模态采用独立超图塔 $g_{\theta_m}$；以下省略模态上标：

$$
\begin{aligned}
Z^{(0)}&=X_{\mathcal H},\qquad
A_{\mathcal H}=D_V^{-1/2}BD_E^{-1}B^\top D_V^{-1/2},\\
U^{(\ell)}&=\sigma_\ell\!\left(\operatorname{LN}_\ell
(Z^{(\ell-1)}W_\ell+\mathbf1b_\ell^\top)\right),\\
E^{(\ell)}&=D_E^{-1}B^\top D_V^{-1/2}U^{(\ell)},\\
Z^{(\ell)}&=U^{(\ell)}+A_{\mathcal H}U^{(\ell)},\qquad
(Z^m,E^m)=g_{\theta_m}(X_{\mathcal H}^m,B^m),\\
\sigma_\ell&=
\begin{cases}
\operatorname{Dropout}\circ\operatorname{GELU},&\ell<L,\\
\operatorname{Identity},&\ell=L.
\end{cases}
\end{aligned}
$$

### 三项对比目标

$s_{\psi_m}$ 为可训练轻量模态编码器；$i_e$ 为超边锚点。GA、SA、SN 分别为图锚点、模态锚点与模态邻域均值：

$$
\begin{aligned}
a_i^m&=\operatorname{Pool}_m(s_{\psi_m}(r_i^m)),\\
\mathrm{GA}_e^m&=Z_{i_e}^m,\qquad \mathrm{SA}_e^m=a_{i_e}^m,\\
\mathrm{SN}_e^m&=\frac1{|e\setminus\{i_e\}|}
\sum_{j\in e\setminus\{i_e\}}a_j^m.
\end{aligned}
$$

对批内配对表示 $P=(p_e),Q=(q_e)$，同一全局物品 ID 的重复锚点均为正样本，$\mathcal P_e=\{j:i_j=i_e\}$：

$$
\begin{aligned}
\ell(P\to Q)&=-\frac1n\sum_{e=1}^{n}
\log\frac{\sum_{j\in\mathcal P_e}\exp(\cos(p_e,q_j)/\tau_g)}
{\sum_{j=1}^{n}\exp(\cos(p_e,q_j)/\tau_g)},\\
\mathcal C(P,Q)&=\tfrac12[\ell(P\to Q)+\ell(Q\to P)].
\end{aligned}
$$

$$
\boxed{\mathcal L_{\mathrm{ground}}^m
=\lambda_1\mathcal C(\mathrm{GA}^m,\mathrm{SA}^m)
+\lambda_2\mathcal C(\mathrm{GA}^m,\mathrm{SN}^m)
+\lambda_3\mathcal C(\mathrm{SA}^m,\mathrm{SN}^m)}
$$

$$
(\theta_m^*,\psi_m^*)=\arg\min_{\theta_m,\psi_m}
\mathbb E_{\mathcal H^m}[\mathcal L_{\mathrm{ground}}^m],
\quad\forall m\in\mathcal M;
\qquad(\lambda_1,\lambda_2,\lambda_3)=(1,3,3),\quad\tau_g=0.07.
$$

四模态独立预训练；涉及 SN 的目标只使用非空邻域。第三项仅直接优化模态编码器，前两项同时优化模态编码器与超图塔。

## 2. CRS

### 超图 tokens 与共享 backbone

以历史物品检索局部超图，分别投影节点与超边；图 token 经模态条件 MoE 适配，$v_m$ 为模态嵌入，$R$ 个专家共享于各模态：

$$
\begin{aligned}
\mathcal H_c^m&=\operatorname{Retrieve}_{K,h}(\mathcal I_c,\mathcal H^m),\\
(Z_c^m,E_c^m)&=g_{\theta_m^*}(X_c^m,B_c^m),\\
T_c^m&=[Z_c^mW_{\mathrm{node}}^m+b_{\mathrm{node}}^m;
E_c^mW_{\mathrm{edge}}^m+b_{\mathrm{edge}}^m],\\
\rho_m(t)&=\operatorname{softmax}(W_\rho[t;v_m]+b_\rho),\\
\widehat t&=t+\gamma_m\sum_{k=1}^{R}\rho_{m,k}(t)\operatorname{MLP}_k(t),\\
\mathcal C_c&=[P_\omega;\operatorname{Emb}(c);
\widehat T_c^{m_1};\cdots;\widehat T_c^{m_4}],\qquad H_c=f_\phi(\mathcal C_c).
\end{aligned}
$$

$P_\omega$ 为可学习 soft prompt；$\mathcal C_c$ 包含模态边界与回复角色。$\mathcal S_c$ 为回复开始前的有效条件位置，排除回复与 padding。

### 模态偏好与得分融合

各模态独立学习查询 $q_0^m$，对共享 backbone 的全部条件隐状态进行 AttnPooling，并将固定离线物品表投影至偏好空间：

$$
\begin{aligned}
q^m&=W_Q^mq_0^m,\qquad k_j^m=W_K^mh_j,\qquad v_j^m=W_V^mh_j,\\
\alpha_j^m&=\operatorname{softmax}_{j\in\mathcal S_c}
\left(\frac{(q^m)^\top k_j^m}{\sqrt{d_a}}\right),\\
p_c^m&=W_O^m\sum_{j\in\mathcal S_c}\alpha_j^mv_j^m,\\
u_c^m&=\operatorname{norm}(\operatorname{MLP}_{u,m}(p_c^m)),\qquad
v_i^m=\operatorname{norm}(W_{i,m}x_i^m),\\
s_{c,i}^m&=\frac{(u_c^m)^\top v_i^m}{\tau_r},\\
\pi_c&=\operatorname{softmax}
\left(\operatorname{MLP}_{\mathrm{router}}([u_c^m]_{m\in\mathcal M})\right),\\
s_{c,i}&=\sum_{m\in\mathcal M}\pi_c^ms_{c,i}^m,\qquad
\widehat{\mathcal R}_c=\operatorname{TopK}_{i\in\mathcal I}s_{c,i}.
\end{aligned}
$$

$\rho_m(t)$ 为图 token 的专家路由，$\pi_c$ 为样本级模态得分路由。推荐融合 logits，并对全物品目录排序。

### 推荐与自回归对话联合目标

$$
\begin{aligned}
\operatorname{CE}(s,y)&=-\log\operatorname{softmax}(s)_y,\\
\mathcal L_{\mathrm{fusion}}&=\operatorname{CE}(s_c,y),\qquad
\mathcal L_{\mathrm{view}}=\frac1{|\mathcal M|}
\sum_{m\in\mathcal M}\operatorname{CE}(s_c^m,y),\\
P_\phi(w_{1:T}\mid\mathcal C_c)&=\prod_{t=1}^{T}
P_\phi(w_t\mid w_{<t},\mathcal C_c),\\
\mathcal L_{\mathrm{conv}}&=-\frac1T\sum_{t=1}^{T}
\log P_\phi(w_t\mid w_{<t},\mathcal C_c).
\end{aligned}
$$

$$
\boxed{\mathcal L_{\mathrm{CRS}}
=\mathcal L_{\mathrm{fusion}}+\eta\mathcal L_{\mathrm{view}}
+\lambda_{\mathrm{conv}}\mathcal L_{\mathrm{conv}}},
\qquad \eta=0.2,\quad\lambda_{\mathrm{conv}}=1,\quad\tau_r=0.07.
$$

CRS 冻结离线特征、预训练图塔与 backbone，训练 soft prompt、节点/超边投影、MoE、AttnPooling、用户/物品投影及推荐路由器。对话损失仅覆盖回复（含 EOS），通过冻结 backbone 回传至输入侧可训练模块；grounding 模态编码器不参与 CRS。

$$
\Theta_{\mathrm{CRS}}^*=\arg\min_{\Theta_{\mathrm{CRS}}}
\mathbb E_{(c,y,w)}[\mathcal L_{\mathrm{CRS}}],
\qquad X^m,\theta_m^*,\phi\ \text{固定}.
$$

$$
\dim(g_{\theta_m}):768\to1024\to1024\to256,
\qquad(d_{\mathrm{LLM}},d_a,d_u)=(2048,256,2048),
\qquad |P_\omega|=20,\quad R=4.
$$

## 3. 评估口径

$$
\mathcal D\in\{\mathrm{ReDial},\mathrm{TG\text{-}ReDial}\},
\qquad
\operatorname{Metrics}=\{mathrm{Recall@K},\mathrm{NDCG@K},\mathrm{MRR@K},
\mathrm{DIST@n},\mathrm{BLEU@n}\},
\quad K\in\{1,10,50\},\ n\in\{1,2,3,4\}.
$$

当前对话指标使用 teacher forcing；自由生成评估使用模型生成前缀：

$$
\widehat w_t^{\mathrm{TF}}=\arg\max_wP_\phi(w\mid w_{<t}^{\mathrm{gold}},\mathcal C_c),
\qquad
\widehat w_t^{\mathrm{AR}}\sim P_\phi(w\mid\widehat w_{<t}^{\mathrm{AR}},\mathcal C_c).
$$

# 后续实验接口：

主实验分为：多模态消融实验(only-*但模态+full（txt,img,ado,vdo，full为主要模态)），LLM backbone实验（Qwen2.5-Omni-3B为主backbone），超参数分析

#### 1. MLLM backbone实验，暂时考虑如下模型，需要相似地使用对应模型的“thinker”（我就拿Qwen 2.5 Omni做类比了）

| 官方模型                                                                                                                                                                                                                                                                                              | 参数规模                                             | 在消融中的作用                                            |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------- |
| [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)                                                                                                                                                                                                                                    | 3B 档；完整发布模型约 6B，当前只加载 Thinker         | **主模型，保留已有基线**                            |
| [sensenova/InteractiveOmni-4B](https://huggingface.co/sensenova/InteractiveOmni-4B)                                                                                                                                                                                                                    | 4B 档；完整权重约 6B                                 | 商汤官方四模态模型，覆盖音视频多轮对话                    |
| [openbmb/MiniCPM-o-4\_5](https://huggingface.co/openbmb/MiniCPM-o-4_5)                                                                                                                                                                                                                                 | 语言底座 8B；整模 9B                                 | 主流 Omni 家族，作为较大容量扩展；优先**2×4 HSDP** |
| [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)                                                                                                                                                                                                                      | 语言约 3.09B；整模约 3.8B                            | **主要 VLM 对照**，与现有 Omni 同代、容量接近       |
| [OpenGVLab/InternVL3\_5-2B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF)                                                                                                                                                                                                                     | 语言 2.0B；整模 2.3B                                 | 主流 InternVL 家族，同时提供低容量参考                    |
| [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)                                                                                                                                                                                                                            | 文本模型，3.09B                                      | **必须做的主要文本 LLM 对照**                       |
| [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)                                                                                                                                                                                                            | 文本模型，3B 档                                      | 非 Qwen 文本对照，检验跨家族适配能力                      |
| `llava-hf/llava-1.5-7b-hf`                                                                                                                                                                                                                                                                          | 图文 VLM；Vicuna v1.5 的 7B 档语言模型 + CLIP 视觉塔 | 经典 LLaVA 基线，增加非 Qwen 的多模态语言底座             |

先只做5个，建议Qwen2.5Omni3B,InteractiveOmni-4B,Llava1.5-7b-hf，Qwen2.5VlL-3B-Instrucut,Qwen2.5-3B-Instruct。对于每个backbone,多做两个“full”，一个full相对与我们HyProRec的full（txt,vdo,ado,img），另一个是相对于模型自己的full,比如Llava1.5-7b-hf应该只有文本图像能力，所以可以做他的img+txt双模态

考虑到显卡限制（8 x 3090 24GB），所以稍大的模型需要使用 HSDP，脚本需要支持 DDP/HSDP。标准八卡 CRS 配置使用 `per_device_train_batch_size=1`、`gradient_accumulation_steps=32`，统一 optimizer update 的有效 batch size 为 $1\times32\times8=256$；单卡 smoke 配置可以缩小累积步数，但正式对比必须恢复为 256。

#### 2. 主模型数据集：ReDial以及TG-ReDial

在远程LAB服务器中`/data2/lhf/HoCRS-v3/data/lhf-tg-redial`为初步处理好的TG-REDIAL数据集，但是隶属于不同项目（HoCRS为早期实现，可能有历史遗留的文件，核心文件`embeddings/`是离线特征，` mm/`原始多模态数据，以及movie_info.csv,train/test/valid.json文件，至于什么预计算的超边表不能直接复用。
相似的/data2/lhf/HyProRec/core/data/hocrs2_redial为Redial的数据集，是同一项目，相适配度会高一些。不过重构完代码后还是该重新计算的就重新计算一下。为了节省存储空间占用，可以使用软链。
又或者，可以从/data2/lhf/.DATA软链。建议新的超边表生成脚本、embeddigns离线特征预计算脚本直接写进.DATA文件夹中对应数据集子文件夹中的/scripts中，不用混入模型主代码中

#### 3. 超参数分析

对话损失权重$\lambda = 0.1, 0.3, 1,3$,以view损失权重$\eta = 0,0.1,0.2,0.5$，

具体实践时，建议先做超参数分析实验，确定好能够平衡好对话、推荐表现的损失权重，再进行后续其他实验的。

# 代码修改任务

我们或许不再进行模型架构旁支的探索了，所以重构代码需要将旁支的逻辑、死代码清理一下
对于WANDB（需要更改为SWANLAB） 指标记录CRS任务指标，主指标（推荐对话指标各项损失）都记在train/,eval/,test/这几个Panel Section中，其他的辅助指标，只只记录rec_router以及Attention Pooling中的部分关键指标就好了，放在other_train/ other_eval/ other_test Panel Section中，具体other中只记
- other_*/AttnPooling/
  - {v}-soft_pro-mass （对于模态v的注意力池化的软提示词的注意力总和）
  - {v}-context-mass (对于模态v的注意力池化的上下文历史的注意力总和)
  - {v}-{v\`}-hyg-mass (对于模态v注意力池化的v\`模态试图超图tokens的注意力总合)
- other_*/RecRouter/
  - txt (批次总txt融合权重的平均值)
  - ...（其他模态的同理，我就省略了）

train和val中，记录细分的三项loss，总loss,推荐@1,10,50、对话指标@1,2,3,4即可。
每一个epoch产生4次（包括epoch结束）eval 评测（可以采用step的评测策略，貌似transformers有预留接口）。只保存最后的和最好的（总eval loss最小）的checkpoints。训练结束后，加载最好的模型再进行一次eval评估和test评估（对于对话指标都采用teacher forcing）

对于Grounding任务，每一个epoch最后评测，只需要记录总loss和三项loss的值即可;

然后考虑到W&B不再支持，所以我们的日志记录需要改为SWANLAB.在训练逻辑相关的脚本中通过加载dotenv中的SWANLAB的API KEY登陆对应的linexox账号


# 重构执行计划

重构目标是在 `refactor` 分支中重新组织 HyProRec 的数据、模型、训练和实验记录代码，同时保持本文件方法论中的 grounding、超图 token、推荐融合和对话目标。重构从当前 master 提交 `8fc6861` 建立，现有工作区内容已作为基线提交保存。代码修改不得删除本文件已有的方法论、LAB 操作规范和代码修改规范。

## 1. 代码结构重构

### 1.1 数据接口

统一 ReDial 和 TG-ReDial 的样本为

$$
(c,\mathcal I_c,y,w),
$$

其中 $c$ 为对话历史，$\mathcal I_c$ 为历史物品，$y$ 为目标物品，$w$ 为目标回复。数据层负责生成局部超图、图 token 位置、推荐标签和语言模型标签；训练期间不得反复读取原始多模态文件或重复执行相同的 tokenization。

离线特征与超边表属于数据资产。数据准备脚本放在 `/data2/lhf/.DATA` 对应数据集目录的 `scripts/` 中，模型代码只通过配置和软链访问。ReDial 与 TG-ReDial 的超边表分别重新计算，不直接复用 HoCRS 旧项目的预计算超边表。

### 1.2 Grounding

Grounding 只保留 `txt`、`img`、`ado`、`vdo` 四个独立模态塔和三项对比损失。删除 `co_*` 视图、共现图训练、共现图 checkpoint 合并和只服务于共现图的配置。

每个模态独立选择最佳 grounding checkpoint。Grounding 验证在每个 epoch 结束时执行，只记录总 loss 和 GA-SA、GA-SN、SA-SN 三项 loss。

### 1.3 Backbone adapter

不同 backbone 通过统一接口接入 CRS：

```text
load_tokenizer()
load_model()
get_input_embeddings()
encode_condition()
get_last_hidden_state()
compute_lm_logits()
prepare_generation_cache()
```

首批适配模型为 Qwen2.5-Omni-3B、InteractiveOmni-4B、LLaVA-1.5-7B、Qwen2.5-VL-3B-Instruct、Qwen2.5-3B-Instruct。Omni 模型使用 thinker 或对应的文本生成子模块；不同模型的 tokenizer、chat template、输入 embedding、hidden state 和 generation cache 差异只能位于 adapter 中，不能散落在训练逻辑中。

每个 backbone 做两个 full 设置：HyProRec full 使用 `txt,img,ado,vdo`；模型自身 full 按原生能力使用四模态、`txt,img` 或 `txt`。图 token 仍由 HyProRec 的离线特征和超图塔提供，CRS 不重复向 backbone 原生视觉、音频或视频编码器输入原始媒体。

### 1.4 CRS 主路径

CRS 只保留以下结构：四个 grounding 超图塔、节点/超边投影、GraphTokenMoE、共享 backbone、四组 AttnPooling、全目录物品评分、推荐 router 和语言模型 head。删除 BPR、router balance、旧版 decoupled CRS、历史推荐头、共现图和未使用的实验旁支。

CRS 默认冻结离线特征、grounding 超图塔和 backbone，只训练 soft prompt、节点/超边投影、GraphTokenMoE、AttnPooling、用户/物品投影及推荐 router。推荐与对话仍使用本文件前述联合损失和指标定义。

## 2. SwanLab 记录要求

训练代码从 `.env` 读取 `SWANLAB_API_KEY`，可选读取 `SWANLAB_API_HOST`，不得把凭证写入代码、配置或提交。SwanLab 项目名为 `HyProRec`。

run 名使用 `only-*` 或 `full` 表示模态组合；tags 必须包含数据集、backbone、模态组合、$\lambda$、$\eta$ 和 seed。主指标写入 `train/`、`eval/`、`test/`，包括总 loss、各项 loss、Recall/MRR/NDCG@1/10/50、BLEU/DIST@1/2/3/4。辅助指标只写入 `other_train/`、`other_eval/`、`other_test/`：

```text
AttnPooling/{view}-soft-prompt-mass
AttnPooling/{view}-context-mass
AttnPooling/{view}-{source_view}-hypergraph-mass
RecRouter/{view}
```

每个 CRS epoch 进行两次 eval，验证点为该 epoch optimizer update 数的中点和末点。只保留 last checkpoint 和 eval 总 loss 最小的 best checkpoint。训练结束后加载 best checkpoint，重新执行一次 eval 和 test；对话指标使用 teacher forcing。Grounding 每个 epoch 结束时验证一次。训练阶段只记录优化所需的 loss；attention、GraphMoE 和 recommendation-router 辅助指标仅在 eval/test 阶段计算和记录。

## 3. 训练和推理速度研究

重构前后使用相同硬件、数据、seed、batch、序列长度和模态配置记录 step time、tokens/s、data wait、GPU 利用率、峰值显存、eval/test 时间及任务指标。每项优化必须有局部 benchmark 和端到端回归结果。

优先检查以下路径：

1. collator 中 history 的 tokenizer-decode-tokenizer 往返；
2. 每个样本、每个视图重复构造局部超图；
3. GraphTokenMoE 中逐 expert 的 Python 循环；
4. 每次 forward 请求完整 hidden states；
5. 冻结 backbone 仍启用 gradient checkpointing 的收益与代价；3B CRS 默认关闭，只有显存不足时才重新开启；
6. 每个训练 step 计算 attention 和 GraphMoE 诊断指标；训练阶段默认关闭，eval/test 阶段保留；
7. generation 中重复注入图 token、未复用 past key values；
8. 评估中保存完整 vocabulary logits 和重复解码；
9. 全目录物品表的重复投影和数据搬运。

优先实现缓存 tokenized history、局部超图和位置 mask，批量化 GraphTokenMoE，减少 hidden state 保存，优化 generation cache 和评估诊断。不得用 `torch.no_grad()` 阻断 soft prompt 或图 token 的输入梯度；`torch.compile`、Flash Attention 和 fused kernel 只有在五类 backbone 都能通过回归后才使用。

性能验收目标为训练 tokens/s 提升至少 30%、峰值显存不增加、generation latency 不回退，且 Recall、NDCG、MRR、BLEU、DIST 与 master 的相对差异不超过 1%。

## 4. LAB 部署和 smoke 验证

本地提交并推送 `origin/refactor` 后，在 LAB 使用 `~/lhf/init.sh` 初始化环境，在 `/data2/lhf/HyProRec/refactor` 建立 worktree。新 worktree 不得修改 `/data2/lhf/HyProRec/core` 的现有脏工作区。

LAB 上优先复用 `$HF_HOME` 已有模型缓存，不重复下载权重。五个 backbone 均需执行 config、tokenizer、单 batch forward 和单步 generation smoke；同时验证 hidden size、logits shape、graph token 注入和 checkpoint 恢复。数据集通过 `/data2/lhf/.DATA` 或现有数据目录软链提供，所有新增文件写入 `/data2/lhf`。

重构完成的验收顺序为：单元测试、grounding 单 batch、CRS 单 batch、五个 backbone smoke、固定小数据集回归、完整 eval/test、速度对比和 LAB 远程运行检查。



## 5. 细化执行任务

以下任务按依赖顺序执行；每项完成后保留可复现命令和结果，不以“能启动”代替验收：

1. **R0 基线与环境**：冻结 `master` 基线和 `refactor` 分支，记录 commit、依赖版本、GPU、seed、数据切分和当前指标；本地工作区与 LAB worktree 分离，禁止触碰 LAB 既有 `core` 脏工作区。
2. **R1 数据资产**：统一 ReDial/TG-ReDial 的样本字段和 split 语义；为每个数据集分别生成四模态离线特征、超边表和缓存索引；模型仓库只保留读取接口，数据准备脚本写入 `/data2/lhf/.DATA/<dataset>/scripts/`，通过软链接入实验。
3. **R2 Grounding**：删除共现图入口和遗留合并脚本；逐模态训练、验证并保存最佳塔；checkpoint 必须能被 CRS 逐塔恢复，验证日志只包含四项 grounding loss。
4. **R3 Backbone adapter**：先实现统一 tokenizer、embedding、decoder hidden、LM logits 和 generation-cache 接口，再逐个接入五类 backbone；每个 adapter 先通过 config/tokenizer/forward/generation 四级 smoke，再允许进入训练脚本。
5. **R4 CRS 主路径**：固定 grounding 与 backbone，按四模态超图 token → MoE → backbone → AttnPooling → 全目录评分 → router/LM 的顺序重构；训练路径不保留 BPR、balance、co 或旧推荐头的参数、分支和日志。
6. **R5 训练与记录**：支持普通 DDP 和 HSDP；以 optimizer update 为单位切分每 epoch 的两次 eval；CRS 默认关闭 gradient checkpointing、关闭 DDP unused-parameter 搜索，DataLoader 使用多 worker 和 pinned memory；只保留 best/last；训练后重新跑 best-eval 与 test；SwanLab 的主指标和辅助指标严格按本文件 panel 命名。
7. **R6 性能实验**：为 tokenization、局部超图、MoE、hidden-state、checkpoint、diagnostics、generation cache 和全目录投影分别记录局部 benchmark；再做固定小数据端到端对比，只有指标、显存和 generation latency 均满足约束时才保留优化。
8. **R7 验收与部署**：依次执行本地单测/compile、grounding 单 batch、CRS 单 batch、backbone smoke、固定回归、完整 eval/test 和速度报告；将通过验证的 commit 用 bundle 同步到 LAB worktree，并记录软链、模型路径和运行命令。

当前已验证的性能实现是：`HoCRSBaseModel.encode` 直接调用 backbone decoder，默认只保留最后 hidden state；有 labels 时只对 shifted 后的有效回复位置投影词表并计算 causal loss，评估阶段返回这些位置的 teacher-forced argmax，训练态不保留词表 logits；无 labels 的 generation 仍返回完整 LM logits 并复用 past key values。该实现必须继续通过数值等价和 cache generation 回归，不能仅凭单项 micro-benchmark 宣称达到总体加速目标。

# 代码修改规范

基本原则：在运算逻辑密集的区域（比如modeling_*.py 等建模的代码）尽量不要写防御性代码。防御性代码会打断阅读代码、了解方法论的思路，所以不要在主要方法论的代码中插入防御性代码。对于运算逻辑密集的区域如果实在需要，只考虑不带额外的错误提示信息的assert语句
代码语句结构、组织需要简洁明了，在相近代码长度下，优先考虑更对称（代码长度相近、语句结构形似）的写法
代码文件权责分明。

不要把多个运算步骤写为一行，这样会导致某一行很长，且存在复杂的嵌套关系，不容易阅读。此外，也尽量减少函数括号换行后出现一个超长参数+一个超短参数，这样极其不对称，观感很难受。在保证运算逻辑不变的前提下，可以将运算拆成多部，中间变量可以使用相同变量名。也可以使用临时变量（变量名最好有实际意义，但需要简介，如果中间临时变量名跟主要变量名一样为了精确而过度复杂冗长，也会导致阅读疲劳）缓存算式结果，用于后续算式计算，避免一个算式太长。

- bad example

```Python
  def forward(
      self, features: torch.Tensor, view: str, return_diagnostics: bool = False
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
      view_embedding = self.view_embeddings[self.view_to_index[view]].expand(features.size(0), -1)
      weights = F.softmax(
        self.router(torch.cat((features, view_embedding.to(features.dtype)), dim=-1).float()),
        dim=-1
      )
      outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
      delta = (weights.to(outputs.dtype).unsqueeze(-1) * outputs).sum(dim=1)
      transformed = (
          features
          + self.residual_scales[self.view_to_index[view]].to(features.dtype) * delta
      )
      return (transformed, weights) if return_diagnostics else transformed
```

- to be better:

  ```Python
   def forward(
      self,
      features: torch.Tensor,
      view: str,
      return_diagnostics: bool = False
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

      view_index = self.view_to_index[view]
      view_embedding = self.view_embeddings[view_index]
      view_embedding = view_embedding.expand(features.size(0), -1)
      router_input = torch.cat((features, view_embedding.to(features.dtype)), dim=-1)
      router_input = router_input.float()
      outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
      weights = F.softmax(self.router(router_input), dim=-1)
      delta = (weights.to(outputs.dtype).unsqueeze(-1) * outputs).sum(dim=1)
      scale = self.residual_scales[view_index].to(features.dtype)
      transformed = features + scale * delta

      if return_diagnostics:
          return transformed, weights
      else:
          return transformed
  ```

可以使用爱因斯坦表达式简化一些代数运算，比如：

- bad example：
- ```Python
  outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
  weights = F.softmax(self.router(router_input), dim=-1)
  delta = (weights.to(outputs.dtype).unsqueeze(-1) * outputs).sum(dim=1)
  ```
- better

```Python
outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
weights = F.softmax(self.router(router_input), dim=-1)
delta = torch.einsum("be,beh->bh", weights.to(outputs.dtype), outputs)
```
