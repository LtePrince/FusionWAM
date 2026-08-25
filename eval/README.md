# 验收:剂量协议(不是分内成功率)

铁律来源:一个分内 99.3% 的模型在 10cm 位移下只有 28.7%。
FusionWAM 的一切训练产物按以下流程验收。

## 环境与工件

- 评测代码已 vendor:`displacement_eval.py`(协议入口)+
  `eval_libero_single.py`(推理助手)+ `libero_utils.py`(env 工具)。
- 需自装外部 sim 栈:LIBERO(github.com/Lifelong-Robot-Learning/LIBERO)+
  robosuite + MuJoCo,EGL 渲染(`MUJOCO_GL=egl`)。
- 模型构建与训练同款:joint 任务 + `create_fusion_model` + stage1.yaml 的
  `vlm_path`;`skip_dit_load_from_pretrain=false`(stage-1 档只存可训部分,
  冻结视频塔从 Wan2.2 底座重载);经 `FusionWAM.load_checkpoint` 加载(adapter
  缺失即报错)。推理经 `vlm_prompt` 传入 DEFAULT_PROMPT 格式化的指令文本
  (底层 `prompt` 与预计算 T5 `context` 互斥);挂了 VLM 却无文本 = 硬错误。
- 数据工件:归一化统计取 checkpoint 所在 run 目录的 `dataset_stats.json`
  (训练自算并写入 `runs/<run>/`,即策略实际使用的统计;勿换成 release 版);
  `data/text_embeds_cache/libero`(precompute_text_embeds.py 的产物)。

## 运行

```bash
MUJOCO_GL=egl .venv/bin/python eval/displacement_eval.py \
    --ckpt <weights checkpoint> --tasks 0-4 --trials 6 \
    --doses 0,0.04,0.07,0.10,0.15 --out results/dose.json
```

协议:init → settle 5 → 目标物按 seeded 随机方向位移 dose → settle 3 →
策略执行(BDDL 判成功);seed 族 7000+trial*10+d_i 与全部历史数字配对可比。
**0 剂量对照必跑**(协议/约定虫探测器;历史上抓过三只)。
`--mask-proprio` 为诊断用(置零绝对 eef 位置维)。

## 判读标准(打包时预注册)

判决格 = 10cm:
- ≥60%:VLM 语义源被真实消费,进 stage 2;
- 45-60%:部分消费,先做源注意力熵诊断再决定;
- ≤45%(≈ 最佳 6B-LoRA 46.7):融合无效——检查 source dropout 是否在跑、
  LayerScale 是否离开初始值(注意力质量 ~0 = VLM 形同虚设)。

对照锚(同协议历史数字):base WAM 100/88/40/20(0/4/7/10cm)、
最佳 6B-LoRA 46.7@10、π0.5 73.3@10、枚举核心 GT 上界 93.3@10
(= oracle 执行器天花板的 100%;15cm 格天花板 77%,报
feasibility-adjusted)。
