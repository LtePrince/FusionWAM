# 验收:剂量协议(不是分内成功率)

战役铁律:分内成功率骗过我们一次(99.3% 的模型在 10cm 位移下 28.7%)。
FusionWAM 的一切训练产物按以下流程验收:

1. 评测脚本已 vendor 在本目录(`eval_relcond.py` + `libero_utils.py`);
   需自装 sim 栈:LIBERO(github.com/Lifelong-Robot-Learning/LIBERO)+
   robosuite + MuJoCo,EGL 渲染(`MUJOCO_GL=egl`)。anchor-detector 相关
   flags 属实验室内部实验,已禁用;
2. 剂量扫描:tasks 0-4 × trials 6 × doses {0,4,7,10,15}cm,legacy seed 族
   (7000+trial*10+d_i)保证与战役所有历史数字 seed 配对可比;
3. **0 剂量对照必跑**(协议前缀/约定虫探测器,战役抓过三只);
4. 15cm 格报 feasibility-adjusted(oracle 执行器天花板 77%);
5. 对照锚:base WAM 20/88/40/20(0/4/7/10cm 顺序为 100/88/40/20)、
   最佳 6B-LoRA 46.7@10、π0.5 73.3@10、枚举核心 GT 上界 93.3@10。

判读标准(预注册于打包时):stage-1 FusionWAM 的判决格是 10cm——
- ≥60%:VLM 语义源被真实消费,进 stage 2;
- 45-60%:部分消费,先做源注意力熵诊断再决定;
- ≤45%(≈ 最佳 6B-LoRA):融合无效,检查 source dropout 是否真的在跑、
  layerscale 是否过小(注意力质量 ~0 = VLM 形同虚设)。
