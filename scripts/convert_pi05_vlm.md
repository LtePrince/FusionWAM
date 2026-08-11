# (可选)π0.5 微调版 PaliGemma → HF 权重转换指引

**先跑判决实验再决定是否做这份工**:用实验机的 Stage-A 冻结探针协议
(FastWAM-LoRA `experiments/cross_embodiment/probe_anchor.py`)分别指向
base PaliGemma 与 π0.5 版特征,测位移场景 OOD 位置读出。若 base 版探针
误差已 ≤2cm(OOD),转换工作可整体跳过。

## 转换要点(openpi JAX → HF PyTorch)

1. 源:openpi 的 π0.5 checkpoint(orbax 格式),PaliGemma 参数在
   `params/PaliGemma/...` 子树(img/ 是 SigLIP,llm/ 是 Gemma)。
2. 命名映射:openpi 的 `llm/layers/attn/q_einsum` 等 einsum 权重需 reshape+
   transpose 成 HF `model.language_model.layers.N.self_attn.q_proj.weight`;
   SigLIP 侧 `img/Transformer/encoderblock_N/...` → `vision_tower.…`。
   参考实现:HF hub 上 `google/paligemma-3b-pt-224` 的原始转换脚本
   (transformers 仓库 `src/transformers/models/paligemma/convert_*.py`)
   反向套用即可,注意 openpi 对 Gemma 用了 RMSNorm zero-centered 约定
   (+1 offset)——转换时给 norm 权重 +1。
3. 校验:同一张图+prompt,JAX 与 HF 前向的最后层 hidden 余弦 ≥0.999。
4. 产物放 `checkpoints/paligemma-pi05/`,配置里换 `fusion.vlm_path`。

工作量估计:1-2 天(含数值对齐调试)。这就是为什么先跑探针。
