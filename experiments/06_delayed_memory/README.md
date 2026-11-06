# 06 · 遮挡后的属性保持与记忆更新

[返回仓库首页](../../README.md) · [上一轮：表示与边界](../05_boundary_state/README.md)

**实验已启动，结果待运行完成。** 本轮检验固定大小的递归状态能否跨越无信息间隔保留视觉属性，并在新观测出现后更新记忆。

## 探究问题

物体进入遮挡区域后，最近视频块逐渐不再包含颜色和形状。保留第一段线索能否解决延迟回忆？当外部观测提示物体属性已改变时，递归状态能否兼顾保持与更新？

## 数据与任务

V3 程序场景由灰色背景、中央遮挡物和运动物体组成，属性为四种颜色与圆形、方形、三角形的 12 种组合。每块 16 帧、192×192。物体在 cue 中进入遮挡物，在 return 中从右侧出现。

| 任务 | 外部观测 | 生成目标 |
|---|---|---|
| keep | 一段属性线索 cue | 若干 hidden 块，再生成保持原属性的 return |
| update | 第一段 cue，再观察属性改变后的第二段 cue | 若干 hidden 块，再生成最新属性的 return |

第二段 cue 是外部观测，不是生成目标。每个场景有两个属性不同的配对变体，背景、几何和轨迹相同；update 配对还共享第一段 cue，其初始属性从排除双方最终属性的其余 10 类中均匀抽取。配对的差异由第二次观测引入。

中间 hidden 块只有静态背景与遮挡物。同一场景所有间隔和两个变体复用同一个 hidden latent 张量。场景随机数、属性分配和延迟分配独立，完整场景对留在同一划分。

冻结本地 Wan VAE。各块独立编码“重复首帧＋16 帧”，丢弃第一个 latent 时间片，得到 `16×4×24×24`，按 VAE 配置的均值和标准差归一化；块间不共享因果缓存。生成解码使用生成 latent 自身的首片补锚，不使用真实 return 首帧。

## 等预算对照

| 组别 | 16×256 表示的更新方式 | FP16 持久条件 |
|---|---|---:|
| recent | 只编码最近一块，不读取旧状态 | 8 KiB 表示＋18 KiB boundary |
| first | 保存第一段 cue 的表示，后续保持不变 | 8 KiB 表示＋18 KiB boundary |
| memory | 当前块与旧状态经可学习门控递归更新 | 8 KiB 表示＋18 KiB boundary |

三组使用相同参数结构和初始化。boundary 始终是最近一块的最后一个 latent 时间片，三组均更新。阶段标记对所有组相同，只告诉模型当前生成 hidden 还是 return，不包含属性，也不进入记忆更新。属性标签仅用于评估。

## 训练与评估

训练／验证／测试分别为 1,000／200／300 个场景对，keep 与 update 平衡。训练延迟为 1、2、3 块，测试在相同场景上评估 1、3、6 块。首轮测试按 metadata 中的任务与最终属性配对分层，确定性选取 **60 个完整场景对**，不代表全量 300 对结果。

三个训练种子为 42、43、44，每个模型训练 6,000 步，batch 为 8 个 episode。训练仅使用 flow matching，hidden 和 return 目标各占一半；三组共享样本、噪声和 dropout。评估使用 32 步 Euler，初始噪声按场景对共享，同时在各组间保持一致。

分别报告真实 hidden 历史与闭环生成 hidden 历史下的 presence、color、shape、完整属性准确率和目标掩码 IoU。颜色与形状指标计入未出现物体的失败。memory 另在真实历史条件下进行状态置零和配对 donor 状态交换，记录输出是否转向 donor 的最终属性。报告逐种子结果、种子间波动及场景对配对比较。

资源测量每次只加载一个模型，batch 1，使用 CUDA 同步计时，分别记录 cue 编码、状态更新和采样耗时，以及首调用和后续调用表现。实际持久状态按底层 storage 字节计数，同时报告 peak allocated／reserved；VAE 解码单独测量，训练时间另记。三组都是固定历史缓存，本轮不据此声称相对无限 KV 历史的加速。

## 运行

在仓库根目录执行：

```bash
python experiments/06_delayed_memory/data.py --root . --device cuda

for seed in 42 43 44; do
  python experiments/06_delayed_memory/train.py --root . --seed "$seed" --steps 6000
done

python experiments/06_delayed_memory/evaluate.py --root . --split test --pairs 60 --seeds 42 43 44 --delays 1 3 6
python experiments/06_delayed_memory/sample.py --root . --seed 42 --delays 3 6
python experiments/06_delayed_memory/resources.py --root . --seed 42 --decode
python experiments/06_delayed_memory/analyze.py --root .
```

也可后台串行执行数据准备、验证集重建标定、三种子训练、测试、资源测量和汇总：

```bash
mkdir -p logs
nohup python -u experiments/06_delayed_memory/run_pipeline.py --root . > logs/delay_pipeline.log 2>&1 < /dev/null &
```

数据缓存及预览位于 `cache/delayed_memory/`；权重、训练记录位于 `runs/delayed_memory_seed{seed}/`，训练恢复加 `--resume`。测试逐例指标、配置和样例位于 `results/delayed_memory/test/`；资源记录位于种子运行目录的 `resources/`；汇总表和图写入本目录的 `results/`。流水线状态及各阶段日志位于 `logs/`。

## 实验结果

结果待运行完成。固定两对场景、两档间隔的六面板视频及帧对照将保存到 `results/samples/`；统计脚本同时归档压缩后的完整逐案例指标，可离线重算。
