# FusionWAM 数据准备

三个数据源,重要性排序。所有管线在实验机 `WAM/FastWAM-LoRA/` 已存在并验证过,
本文只写"迁移时拷什么、改哪一行"。

## 1. LIBERO 四元组(基线,必需)

lerobot 格式(帧/语言/动作/本体感),FastWAM-LoRA `data/libero_lerobot/` 直接
rsync;或从原始 HDF5 重建(`experiments/libero/` 转换脚本)。
**约定检查单**(逐条核对,每条都对应一次历史事故):
- 轴角 +π 覆盖(dim0 非负);
- 夹爪 lerobot {0,1}(1=open),env 原生 {-1(open),+1(close)},转换 `(1-a)/2`;
- norm stats 随 checkpoint 走,不随数据集走;
- 图像 224×224,agentview 需 `[::-1,::-1]` 翻转(LIBERO 渲染约定)。

## 2. 枚举渲染集(支撑进预训练的关键,强烈建议)

实验机 `experiments/envelope/enum_dataset.py` 是状态版(无渲染)。渲染版改动:
1. `run_scripted()` 已有 `frames` 参数收 agentview;再加 wrist 相机同法;
2. 录制侧换成 `build_aug_dataset.py` 的 `LerobotWriter`(视频编码/元数据全现成),
   每步存 `capture_step(obs)` 而非裸 state;
3. 产能账:~1,700 集 × 2 相机 × 224² ≈ 35G 视频,生成 ~10h(单 EGL GPU)。
   位移半径 r≤0.20、2cm 网格与 P2(c) 相同;**校准 demo 自动选择保留**
   (task2-4 靠它从 0% 修到 86-98%)。

## 3. 免费子任务标签(π0.5 式分层要用)

枚举生成器的 primitive 阶段即标签:approach/grasp/transport/release 的
step 区间在 `script_waypoints()` 返回值里逐段可知——在 LerobotWriter 的
episode 元数据里加 `subtask_spans` 字段即可。人工成本为零,这是脚本生成
路线独有的红利。

## 4. (仅阶段 2)web VL 混合集

解冻 PaliGemma 时按 π0.5 配方混 web 视觉-语言数据防绑定塌缩;
比例起点 robot:web = 1:1,依据战役 P1 判决(宽包络来自预训练多样性),
塌缩监测 = 训练中每 N 步跑一次剂量协议 4/10cm 两格(眼睛不能只盯 loss)。
