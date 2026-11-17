# 深度信念网络（DBN）MNIST 分类复现

堆叠 RBM 逐层贪婪预训练 + 有监督微调，对比「无预训练 / 有预训练 / 预训练 + 输入二值化」三种策略对分类效果的影响。

代码基于开源项目 [AmanPriyanshu/Deep-Belief-Networks-in-PyTorch](https://github.com/AmanPriyanshu/Deep-Belief-Networks-in-PyTorch) 完成复现与对比实验。

## 文件说明

- `DBN.py` / `RBM.py`：模型与训练（CD-k 对比散度）
- `MNIST_DBN_classifier_example.py`：主实验
- `*_classifier.csv`：三种策略的准确率曲线数据
- `DBN实验报告.md`：完整实验报告
- `images_DBN/`、`images_RBM/`：结果图
- `draw_graphs.py` / `make_dbn_figures.py`：绘图脚本

## 怎么跑

```bash
pip install torch numpy pandas matplotlib
python MNIST_DBN_classifier_example.py
```

## 说明

MNIST 数据集与 `*.pt` 权重未随仓库上传。
