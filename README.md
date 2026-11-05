# DINO-Video-Experiment

**Frozen V-JEPA semantics for next-chunk video denoising**

受 [DINO Forcing](https://arxiv.org/abs/2610.11751) 的语义预测与自条件思路启发，本仓库记录一个视频生成的初步实验：**当模型已经能看到上一视频块的完整 latent 时，额外提供该块的预测语义，能否改善下一块去噪？**

在 UCF101 六类、单个训练种子的验证集上，加入预测语义使平均速度预测 MSE 从 **0.431757** 降至 **0.427852**，相对下降 **0.904%**。这是一个相邻块去噪实验，尚未实现完整的 two-pass 生成器，也没有验证语义替代历史或长视频生成。

## Idea & Method

取连续视频块 A、B，使用冻结的 Wan2.1 causal VAE 得到 `z_A`、`z_B`。冻结的原版 V-JEPA ViT-L/16 单独编码干净的 A，产生时空教师特征 `phi(A)`。

1. **预测历史语义。** 训练一个 139.8 万参数的 3D 卷积预测头 `q(z_A)`，逐位置对齐 `phi(A)`，损失为 `1 - cosine`。教师只看到 A，不读取未来 B。预测头训练结束后冻结。
2. **成对训练去噪器。** 两组使用相同的 3D U-Net、初始化、训练窗口、噪声和时间步。每组约 1,343 万参数，空间通道为 64/128/256，在瓶颈处通过交叉注意力读取 64 个语义 token。

| 实验组 | 历史条件 | 语义条件 |
|---|---|---|
| A · Latent only | 完整的 `z_A` | 全零 token，保留相同网络结构 |
| B · + Predicted semantics | 完整的 `z_A` | 冻结预测头输出 `q(z_A)`，逐 token L2 归一化 |

对下一块进行 flow matching，约定 `t=0` 为纯噪声、`t=1` 为干净 latent：

$$
z_{B,t}=(1-t)\epsilon+t z_B,\qquad v^\star=z_B-\epsilon,\qquad
\mathcal L=\mathbb E\left[\|v_\theta(z_{B,t},t,z_A,h)-v^\star\|_2^2\right].
$$

因为 `q(z_A)` 由同一段历史计算，两组的原始历史信息相同。实验测量的是：预训练语义监督能否让生成器更容易利用这段历史。

## Repository Structure

```text
DINO-Video-Experiment/
├── README.md
├── requirements.txt
├── scripts/
│   ├── download_assets.sh       # 下载模型、官方代码与 UCF101 六类数据
│   ├── prepare_cache.py         # 划分视频、提取 VAE / V-JEPA 缓存
│   ├── train_semantic_head.py   # 训练历史语义预测头
│   ├── denoiser_model.py        # 3D U-Net 与语义交叉注意力
│   ├── denoiser_stats.py        # 配对统计与来源组 bootstrap
│   ├── train_denoiser_ab.py     # A/B 训练与条件干预
│   └── plot_denoiser_ab.py      # 绘制验证曲线和误差区间
├── assets/
│   └── ab_denoising_seed42.png
└── results/
    ├── semantic_head_seed42/    # 配置、验证曲线和最佳指标
    └── denoiser_ab_seed42/      # 配置、曲线、逐窗口误差和最终统计
```

运行时生成的 `models/`、`data/`、`third_party/`、`cache/`、`runs/` 和 `logs/` 均不进入 Git。公开的 `results/` 只保留复核本轮实验所需的小文件；其配置路径已改为仓库相对路径。

## Installation

实测环境：Linux、Python 3.12.3、PyTorch 2.8.0 + CUDA 12.8、RTX 4080 SUPER（服务器分配 32 GB 显存）。以下命令在 Linux / AutoDL 上执行。

```bash
git clone https://github.com/johnmayer123-bluser/DINO-Video-Experiment.git
cd DINO-Video-Experiment

# AutoDL root 用户；普通用户可在 apt-get 前加 sudo。
apt-get update
apt-get install -y git curl unzip unrar

# 已有实验环境时可直接使用现有 Python。
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Download Models & Dataset

```bash
# 在仓库根目录执行；支持续传，已有的完整资源会跳过。
bash scripts/download_assets.sh
```

| 资源 | 下载内容与来源 | 本地位置 |
|---|---|---|
| Wan2.1 VAE | [Wan-AI 官方 Diffusers 权重](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tree/main/vae)，只取 VAE 配置和权重，约 508 MB | `models/wan-vae/` |
| V-JEPA ViT-L/16 | [Meta 官方模型](https://github.com/facebookresearch/jepa#model-zoo)，使用 `target_encoder`，约 5.14 GB | `models/vjepa/vitl16.pth.tar` |
| V-JEPA 代码 | [facebookresearch/jepa](https://github.com/facebookresearch/jepa)，固定本次实验的 commit | `third_party/jepa/` |
| UCF101 | [官方数据与划分](https://www.crcv.ucf.edu/research/data-sets/ucf101/)，整包约 6.93 GB，仅解压下面六类 | `data/UCF-101/`、`data/ucfTrainTestlist/` |

下载脚本固定 Wan revision `dd9313515fdad8a9fcef162db68309090805ed30`，V-JEPA 代码 commit `51c59d518fc63c08464af6de585f78ac0c7ed4d5`，并校验两项模型权重的 SHA256。

六类为 `BasketballDunk`、`GolfSwing`、`LongJump`、`TennisSwing`、`VolleyballSpiking`、`Diving`。官方 UCF101 提供完整 RAR，脚本先下载整包，再选择性解压六类。确认视频已解压后，可释放压缩包占用：

```bash
rm data/UCF101.rar
```

本次实验只使用 Wan 的 **VAE**，不需要下载 Wan 的扩散生成器或文本编码器。模型、数据和第三方代码分别遵循其上游许可与使用条款。

## Data & Preprocessing

使用 UCF101 官方 split 1；从官方训练集按“类别＋来源组”留出约 20% 作为验证集，固定划分种子 42。同组视频及其窗口不会跨集合。测试集保留，不用于本次结果。

| 划分 | 原视频数 | 33 帧窗口数 |
|---|---:|---:|
| Train | 415 | 1,624 |
| Validation | 120 | 475 |
| Test（未评估） | 196 | 759 |

六类原有 833 个视频，102 个视频在 12 fps 下不足 33 帧，最终获得 2,858 个窗口。每个视频最多取 4 个起点。

- 统一采样为 12 fps，每窗口 33 帧：第 0 帧作为 VAE 起始帧；A 为 1–16 帧，B 为 17–32 帧。
- 两路编码采用相同的中心正方形视野。VAE 输入 192×192；V-JEPA 输入 224×224，并使用 ImageNet 通道归一化。
- VAE 连续编码整段，使用后验均值以及权重规定的 latent 标准化。缓存形状为 `[16,9,24,24]`，A 取 `[:,1:5]`，B 取 `[:,5:9]`。
- V-JEPA 对 A、B **分别前向**，单块时空 token 池化到 `[4,4,4,1024]`。历史条件仅来自 A。

```bash
mkdir -p logs
python -u scripts/prepare_cache.py --root . --stage all
```

缓存写入 `cache/ucf6_12fps_192/`，包括划分清单、预处理元数据及逐窗口张量。重复执行会跳过已有缓存。

## Training

以下顺序对应本次实际运行。语义头先训练 1,000 步，再恢复到 3,000 步；A/B 去噪器先各训练 3,000 步，再同时恢复到 6,000 步。

```bash
# 1. 训练语义预测头：q(z_A) -> phi(A)
python -u scripts/train_semantic_head.py --root . \
  --steps 1000 --batch-size 16 --seed 42
python -u scripts/train_semantic_head.py --root . \
  --steps 3000 --batch-size 16 --seed 42 --resume

# 2. 冻结预测头，成对训练 A/B 去噪器
python -u scripts/train_denoiser_ab.py --root . \
  --steps 3000 --batch-size 16 --seed 42
python -u scripts/train_denoiser_ab.py --root . \
  --steps 6000 --batch-size 16 --seed 42 --resume

# 3. 绘制本次运行结果
python scripts/plot_denoiser_ab.py runs/denoiser_ab_seed42 --step 6000
```

预测头和去噪器均使用 AdamW，学习率 `2e-4`、batch 16、BF16。语义头按验证集中心化正负样本相似度差选点，本次最佳为第 2,800 步。A/B 主比较固定使用两组第 6,000 步权重；第 2,000→3,000 步两组验证误差仍各下降超过 2%，因此按既定规则同时续训。

| 输出 | 路径 |
|---|---|
| 冻结语义预测头 | `runs/semantic_head_seed42/best.pt` |
| A/B 两组最终参数 | `runs/denoiser_ab_seed42/weights_step06000.pt`，字典键为 `models.A`、`models.B` |
| A/B 断点（含优化器和随机流） | `runs/denoiser_ab_seed42/last.pt` |
| 逐窗口误差与最终统计 | `runs/denoiser_ab_seed42/errors_step06000.npz`、`summary_step06000.json` |

SSH 断开后仍需运行时，可用 `nohup` 启动某一训练阶段，例如：

```bash
nohup python -u scripts/train_denoiser_ab.py --root . \
  --steps 3000 --batch-size 16 --seed 42 \
  > logs/denoiser_ab.log 2>&1 < /dev/null &
tail -f logs/denoiser_ab.log
```

已有检查点时使用 `--resume`。只评估服务器上的最终 A/B 检查点：

```bash
python -u scripts/train_denoiser_ab.py --root . --eval-only
```

仓库提供指标与误差数据，不分发训练检查点。无需 GPU 或模型即可重画已公开的结果图：

```bash
python scripts/plot_denoiser_ab.py results/denoiser_ab_seed42 --step 6000
```

## Results

### Semantic predictor

| 验证指标 | 结果 |
|---|---:|
| 与教师的逐 token cosine | 0.83885 |
| 固定输出训练均值的 cosine 基线 | 0.71430 |
| 同类视频 9 选 1 检索准确率 | 87.84% |
| 随机检索准确率 | 11.11% |

检索负例来自同类别、不同来源组的 8 个视频；先逐 token 归一化，再减训练集均值。详见 [预测头结果](results/semantic_head_seed42/best_metrics.json)。

### Next-chunk denoising

![Validation curves and paired effects](assets/ab_denoising_seed42.png)

验证固定五档 `t` 与每个窗口的一份高斯噪声。先平均窗口，再按原视频、来源组和类别逐级宏平均。95% 区间通过类别内来源组配对 bootstrap（5,000 次）计算，共 24 个验证来源组。

| t（0=纯噪声，1=干净） | A：latent only ↓ | B：+ semantics ↓ | MSE 相对下降 | 95% 来源组区间 |
|---|---:|---:|---:|---:|
| 0.1 | 0.228637 | 0.228781 | −0.063% | −0.906%～0.788% |
| 0.3 | 0.263176 | 0.260799 | 0.903% | 0.413%～1.355% |
| 0.5 | 0.331763 | 0.325822 | 1.791% | 1.530%～2.076% |
| 0.7 | 0.478778 | 0.470572 | 1.714% | 1.503%～1.938% |
| 0.9 | 0.856430 | 0.853288 | 0.367% | 0.236%～0.492% |
| **五档平均** | **0.431757** | **0.427852** | **0.904%** | **0.695%～1.120%** |

固定 B 的最终权重，只干预语义条件：

| 干预 | 窗口数 | 正确语义相对该干预的 MSE 改善 |
|---|---:|---:|
| 语义置零 | 475 | 1.128% |
| 换成同类、不同来源组视频的语义 | 475 | 1.127% |
| 换成同视频较早且不重叠片段的语义 | 156 | 0.285% |

完整的分组统计、区间及条件干预见 [最终统计](results/denoiser_ab_seed42/summary_step06000.json)。[逐窗口误差](results/denoiser_ab_seed42/errors_step06000.npz) 可与同目录的 `validation_manifest.json` 按 ID 对齐，复核来源组统计。

两组各 6,000 步的训练及周期验证累计约 **11.4 分钟**，PyTorch 记录的峰值已分配 GPU 显存约 **0.73 GiB**。该计时不含模型下载、冻结缓存提取、启动和最终干预/bootstrap 分析；该显存不代表 VAE/V-JEPA 提取阶段的需求。

### Scope

本轮结果支持：在真实完整历史 latent 已存在时，预测语义对下一块速度预测提供了小幅帮助。收益主要出现在 `t=0.5、0.7`，`t=0.1` 尚无明确收益。

当前只有一个训练种子、固定验证噪声与六类开发数据；来源组 bootstrap 区间不包含训练种子或噪声重采样的波动。验证集参与语义头选点及续训决策，测试集未使用。尚未验证语义替代历史、长期记忆、滚动生成、视频感知质量或 V-JEPA 相对于其他压缩方法的独有优势。

## References

- [Dino Forcing Flow Models: Do not denoise what you can predict](https://arxiv.org/abs/2610.11751) · [Code](https://github.com/arijit-hub/dino_forcing)
- [V-JEPA](https://github.com/facebookresearch/jepa)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/)

本仓库是独立探索实验，不是上述工作的官方实现。
