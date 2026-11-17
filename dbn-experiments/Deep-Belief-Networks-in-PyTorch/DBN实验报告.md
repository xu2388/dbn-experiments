# 深度信念网络（DBN）MNIST 手写数字分类实验报告

> 项目：[AmanPriyanshu/Deep-Belief-Networks-in-PyTorch](https://github.com/AmanPriyanshu/Deep-Belief-Networks-in-PyTorch)
> 本地目录：`d:\大三上的资料\假期学习记录\深度信念网络\Deep-Belief-Networks-in-PyTorch`
> 实验日期：2026-08-21
> 硬件：NVIDIA GeForce RTX 4060 Laptop（8.6GB 显存），torch 2.6.0+cu118

---

## 1. 实验目的

用 **深度信念网络（Deep Belief Network, DBN）** 在 **MNIST 手写数字分类**上演示：
1. **逐层贪心预训练（Greedy Layer-wise Pre-training）**：把多个受限玻尔兹曼机（RBM）堆叠，无监督逐层训练。
2. **对比实验**：验证「无预训练」 vs「有预训练」vs「预训练 + 输入二值化」三种策略对分类效果的影响（这正是 DBN 论文的核心思想——无监督预训练为有监督微调提供良好初始点）。

---

## 2. 模型结构

实现方式（[MNIST_DBN_classifier_example.py](MNIST_DBN_classifier_example.py)）：

- **网络拓扑**：`784 → 512 → 128 → 64 → 10`（输入 784\*，4 个隐藏/输出层）。
- **预训练阶段**（[DBN.py 的 train_DBN](DBN.py#L49-L63)）：把结构拆成 4 个 Bernoulli RBM **逐层贪心训练**，每层 RBM：
  - 训练算法：对比散度 **CD-k（k=10）**，SGD，**Adam(lr=0.0005)**
  - `epochs=100`，`batch_size=128`，`early_stopping_patience=10`
- **有监督微调阶段**：把预训练权重初始化一个 MLP（含 Sigmoid 隐层 + Softmax 输出），用 **CrossEntropyLoss + Adam(lr=0.001)** 再训 **5 个 epoch**。
- 数据：仓库内置 CSV（`dataset/MNIST/train.csv` 60000 张 / `test.csv` 10000 张），像素归一化到 [0,1]。

---

## 3. 训练耗时

| 阶段 | 耗时 |
|------|------|
| 4 层 RBM 无监督预训练（每层 100 epoch） | 约 6 分钟 |
| 3 组分类器微调（各 5 epoch） | 约 1.2 分钟 |
| **总计** | **约 7-8 分钟**（GPU） |

全程峰值显存极低（batch 128，MLP 很小），8GB 显存绰绰有余；瓶颈在 Python 层逐 batch 的 CD 循环而非 GPU 算力。

---

## 4. 实验结果

### 4.1 三个模型的测试集最终指标

| 模型 | 最终测试准确率 | 最终测试损失 |
|------|:---:|:---:|
| 无预训练（从头训练 MLP） | **0.9339** | 1.5318 |
| 有预训练（RBM 贪心预训练初始化） | **0.9642** | 1.5002 |
| 预训练 + 输入二值化 | **0.9686** | 1.4938 |

**结论：**
- RBM **逐层预训练**让测试准确率从 **93.4% 提升到 96.4%**（+3 个百分点），充分体现无监督预训练作为良好初始点的作用。
- 在预训练基础上再做**输入二值化**（0/1），进一步跳到 **96.9%**，符合 Bernoulli-Bernoulli RBM 对二值输入的建模假设（原始灰度输入与伯努利采样不完全匹配）。

### 4.2 逐 epoch 训练/测试指标

**① 无预训练**

| epoch | train loss | test loss | train acc | test acc |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.7960 | 1.7966 | 0.707 | 0.706 |
| 2 | 1.6416 | 1.6456 | 0.829 | 0.824 |
| 3 | 1.6201 | 1.6250 | 0.845 | 0.839 |
| 4 | 1.5579 | 1.5665 | 0.920 | 0.910 |
| 5 | 1.5239 | 1.5318 | 0.942 | **0.934** |

**② 有预训练**

| epoch | train loss | test loss | train acc | test acc |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.6250 | 1.6300 | 0.853 | 0.849 |
| 2 | 1.5153 | 1.5247 | 0.956 | 0.947 |
| 3 | 1.4984 | 1.5107 | 0.969 | 0.956 |
| 4 | 1.4904 | 1.5043 | 0.975 | 0.961 |
| 5 | 1.4844 | 1.5002 | 0.980 | **0.964** |

**③ 预训练 + 输入二值化**

| epoch | train loss | test loss | train acc | test acc |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.4824 | 1.4994 | 0.981 | 0.964 |
| 2 | 1.4789 | 1.4970 | 0.984 | 0.967 |
| 3 | 1.4770 | 1.4970 | 0.985 | 0.965 |
| 4 | 1.4747 | 1.4940 | 0.987 | 0.968 |
| 5 | 1.4736 | 1.4938 | 0.988 | **0.969** |

**观察**：有预训练的模型在**第 1 个 epoch 就达到 ~85%**（无预训练仅 ~71%），证明预训练提供了远优于随机初始化的起点，收敛更快、最终更好。

---

## 5. 可视化结果

### 📊 图 1：三种策略的损失与准确率曲线
图片地址：
- `d:\大三上的资料\假期学习记录\深度信念网络\Deep-Belief-Networks-in-PyTorch\results_dbn\acc_loss_curves.png`

（左：交叉熵损失，实线训练/虚线测试；右：准确率。绿线「预训练+二值化」全程最高，蓝线「纯预训练」居中，红线「无预训练」最差。）

### 📊 图 2：DBN 数字重建可视化（0-9 原图 vs 重建）
图片地址：
- `d:\大三上的资料\假期学习记录\深度信念网络\Deep-Belief-Networks-in-PyTorch\results_dbn\digit_reconstruction.png`

（用训练好的 DBN 权重对测试集每个数字抽 1 张做「编码→解码」重建，左列原图 / 右列重建图，展示 RBM/DBN 的无监督生成式重建能力。）

---

## 6. 结论

1. **DBN 逐层贪心预训练有效**：预训练把 MNIST 测试准确率从 93.4% 提升至 96.4%，且前几个 epoch 收敛明显加快。
2. **输入二值化契合 RBM 建模**：进一步到 96.9%，验证了 Bernoulli RBM 对输入分布的假设。
3. 作为生成模型，DBN 的 RBM 层能重建数字（图 2），兼具判别与生成能力，是深度学习无监督预训练思想（深度学习核心基石之一）的直观实验。

---

## 7. 产物文件清单

| 内容 | 绝对路径 |
|------|---------|
| 本报告 | `d:\大三上的资料\假期学习记录\深度信念网络\Deep-Belief-Networks-in-PyTorch\DBN实验报告.md` |
| 三种策略 acc/loss 曲线图 | `...\results_dbn\acc_loss_curves.png` |
| 0-9 数字重建图 | `...\results_dbn\digit_reconstruction.png` |
| 无预训练指标 | `...\DBN_without_pretraining_classifier.csv` |
| 有预训练指标 | `...\DBN_with_pretraining_classifier.csv` |
| 预训练+二值化指标 | `...\DBN_with_pretraining_and_input_binarization_classifier.csv` |
| 训练好的权重 | `...\mnist_trained_dbn.pt`（逐层 RBM 参数）、`...\mnist_trained_dbn_classifier.pt`（分类器） |
| 完整训练脚本 | `...\MNIST_DBN_classifier_example.py` |
| 画图脚本 | `...\make_dbn_figures.py` |