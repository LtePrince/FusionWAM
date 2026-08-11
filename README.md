# FusionWAM:语义-动力学-动作三专家联合训练(迁移包)

> 状态:**迁移包(2026-08-11 打包)**。融合代码按 FastWAM/PaliGemma 真实接口编写,
> 未在本机做过 9B 级端到端 smoke test(2×3090 放不下)——迁移到大机后第一件事
> 是跑 `scripts/smoke_test.py`。本包遵循"代码+配置+数据脚本入库,权重/数据集
> 用下载脚本"的约定。

## 动机(一段话)

2026-08 包络战役(WAM/note/experiment_log/)的三个判决:
①π0.5 的宽空间包络(10cm 位移 73.3% vs FastWAM 28.7%)来自**预训练支撑广度**,
不是架构(π0-base≈FastWAM);②FastWAM 的 6B 视频绑定在微调级预算下不可改写
(增广/强制视觉/双倍预算三度触墙 ~37-47%);③消费机制本身在完整支撑上可饱和
可行性天花板(4.5M 枚举核心 10cm=93.3%=天花板 100%)。
结论:把 **VLM 的语义绑定**(π0.5 型,携带宽包络)与 **视频模型的动力学表征**
(FastWAM 型,协同生成目标值 4-8 分)同时挂给动作去噪器,支撑缺口用
**枚举渲染数据**在联合训练层补——这是三条证据链指向的同一个架构。

## 架构

```
PaliGemma VLM(3B,冻结/阶段2解冻) ──语义 KV──┐
                                              ├──> 动作 DiT(1B,flow matching)
Wan2.2 视频 DiT(5B,FastWAM 权重)──动力学 KV──┘      Q 注意 [VLM ⊕ 视频 ⊕ 动作]
```

- 复用 FastWAM 的 MoT 联合注意力模式(`FastWAMJoint._build_mot_attention_mask`):
  拼接序列 + 分段 mask,各段各自的专家参数。本包把两段扩成三段。
- **阶段 1(本包默认)**:双 backbone 冻结,只训动作专家 + 两个 KV 投影 +
  段嵌入。VLM 与视频段互不注意(不动冻结权重的分布)。
- **阶段 2(大机)**:VLM 替换 umt5 成为视频专家的文本条件源,全 MoT,
  按 π0.5 配方带 web 混合集解冻联合训练。

## 战役教训直接编码进本包的三条设计(不可省略)

1. **源级 dropout**(`fusion.source_dropout`,默认 p=0.3/0.3):训练时随机置零
   视频 KV / VLM KV,强迫每条通路独立携带任务。依据:Gate 1/B2 证明
   "挂上≠用上",P3-B 证明训练期捷径切断有效但要在预训练层做。
2. **VLM 默认冻结**:P1 机制判决 → π0.5 型绑定的宽支撑来自预训练多样性,
   窄分布微调会塌缩。解冻必须配 co-training 混合集(configs/stage2 注释)。
3. **剂量协议验收**:任何训练产物必须过 FastWAM-LoRA
   `experiments/cross_embodiment/eval_relcond.py` 的 0/4/7/10/15cm 扫描
   (含 0 剂量对照与可行性天花板归一),分内成功率不作为验收指标。

## 目录

```
src/fusionwam/fusion_model.py   三专家融合(核心新代码)
src/fusionwam/vlm_adapter.py    PaliGemma 冻结编码器 + KV 投影
src/fusionwam/source_dropout.py 源级 dropout
train.py                        阶段1 训练入口(改自 FastWAM trainer)
scripts/download_weights.sh     权重下载(PaliGemma / FastWAM / umt5)
scripts/convert_pi05_vlm.md     (可选)π0.5 微调版 PaliGemma 的 JAX→HF 转换指引
scripts/prepare_data.md         数据准备:LIBERO 转换 / 枚举重渲染 / 免费子任务标签
scripts/smoke_test.py           迁移后第一步:1 batch 前向+反向
configs/stage1.yaml             冻结双 backbone 的默认配置
eval/README.md                  剂量协议验收流程
```

## 数据计划(重要程度排序)

1. **LIBERO 四元组**(帧/语言/动作/本体感):FastWAM-LoRA `data/` 管线现成。
2. **枚举渲染集**:FastWAM-LoRA `experiments/envelope/enum_dataset.py` 目前
   只录状态;按 `scripts/prepare_data.md` §2 加相机重渲染,得到像素一致的
   枚举密度演示(这是支撑进预训练的关键,DemoGen 密度的上位替代)。
   同一生成器的 primitive 阶段(接近/抓取/运送/释放)= 免费子任务标签(§3)。
3. **web VL 混合集**(仅阶段 2 解冻时):π0.5 配方,防绑定塌缩。

## 硬件预算

阶段 1:3B(冻结,bf16 ≈6G)+ 5B(冻结 ≈10G)+ 1B 训练态(AdamW ≈12G)
+ 激活 ——**建议 ≥2×A100-80G 或 4×A6000**;梯度检查点已在配置里打开。
阶段 2 全参:≥8×A100-80G。2×3090 只能跑阶段 1 的 batch=1 冒烟(勉强)。

## 已知风险与开放问题

- **π0.5 微调版 PaliGemma 是否比 base PaliGemma 好**:未测。判决实验
  (Stage-A 冻结探针指向两者特征,协议在 FastWAM-LoRA
  `experiments/cross_embodiment/probe_anchor.py`)本机可跑,建议迁移前先做——
  若 base 版特征就携带 OOD 位置,省去 JAX 转换的整个工作量。
- 融合注意力的数值尺度(两个 KV 源的范数差)可能需要 per-source LayerScale,
  代码里留了开关(`fusion.source_layerscale`),冒烟时看注意力熵决定。
- 视频专家的 RoPE 频率表与 VLM 位置编码不同源,当前按段独立编码,
  跨段相对位置无定义——这是有意的(动作 Q 对两源的注意应当是内容寻址)。
