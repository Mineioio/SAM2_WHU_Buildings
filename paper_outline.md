# Paper Outline: Zero-Shot Building Segmentation Using SAM 2 on High-Resolution Aerial Imagery



## Title 

- Evaluating SAM 2 for Zero-Shot Building Segmentation in High-Resolution Aerial Imagery: A Comprehensive Benchmark Study


---

## Abstract (200–250 words)

**写什么**：

- **背景**（2句）：Building extraction from remote sensing imagery is important for urban planning and disaster assessment. Foundation models like the Segment Anything Model 2 (SAM 2) offer promising zero-shot segmentation capabilities, but their applicability to remote sensing building extraction remains underexplored.
- **目的**（1句）：This study systematically evaluates SAM 2's performance on aerial building segmentation using the WHU Building Dataset.
- **方法**（2–3句）：We design three groups of experiments: (1) comparison of four SAM 2 model variants (Tiny to Large) under automatic segmentation mode; (2) stratified analysis across five building morphology types (dense, sparse, large-scale, small-scale, and irregular); (3) comparison of three oracle-prompting strategies (single-point, multi-point, and bounding-box) derived from ground-truth annotations.
- **结果**（2–3句）：写实验中最核心的数字，例如 box prompting achieves the highest IoU of X.XX, while automatic mode yields X.XX; SAM 2 performs best on large regular buildings but struggles with dense small buildings.
- **结论**（1句）：Our findings provide practical guidelines for applying SAM 2 to aerial building segmentation and highlight its strengths and limitations in this domain.

**Keywords**: SAM 2, building segmentation, remote sensing, zero-shot segmentation, foundation model, prompt engineering

---

## 1. Introduction (约1000 words)

### 第一段：研究背景与重要性

**写什么**：Building footprint extraction from remote sensing imagery is a fundamental task in urban computing, supporting applications such as urban planning, population estimation, disaster damage assessment, and land use mapping. 引用 [1][2]。

### 第二段：传统方法与深度学习方法

**写什么**：Traditional methods rely on handcrafted features. Deep learning methods (U-Net, DeepLabV3+, HRNet) have achieved high accuracy but require large-scale pixel-level annotations, which are expensive and time-consuming for remote sensing data (a single 512×512 tile takes 5–10 minutes for a professional annotator). 引用 [3][4][5]。

### 第三段：视觉基础模型的兴起

**写什么**：The emergence of vision foundation models, particularly the Segment Anything Model (SAM) and its successor SAM 2, introduces a new paradigm. SAM 2 supports promptable segmentation (points, boxes, masks) and demonstrates strong zero-shot generalization across diverse visual domains. However, SAM 2 was primarily trained on natural images and videos (SA-V dataset with 51K videos), raising the question of whether it generalizes well to the overhead perspective and unique characteristics of aerial remote sensing imagery. 引用 [6][7][8]。

### 第四段：现有研究的不足

**写什么**：Several studies have evaluated SAM (v1) on remote sensing tasks [9][10], but evaluations of SAM 2 in this domain remain limited. Moreover, existing studies often test only one prompting strategy or one model size, lacking a systematic comparison across prompt types, model scales, and building morphologies. 明确指出你论文填补的空白。

### 第五段：本文贡献

**写什么**：列出 3 条贡献——

1. We provide the first systematic evaluation of SAM 2 (including the improved SAM 2.1 checkpoints) on the widely-used WHU Building Dataset, covering four model sizes.
2. We design a stratified evaluation framework that analyzes SAM 2's performance across five representative building morphology types.
3. We compare three oracle-prompting strategies and provide practical guidelines for prompt selection in aerial building segmentation.

---

## 2. Related Work (约800 words)

### 2.1 Building Extraction from Remote Sensing Imagery

**写什么**：简要回顾——

- 早期方法：morphological operations, edge detection, object-based image analysis (OBIA)
- 深度学习方法：FCN → U-Net [3] → DeepLabV3+ [4] → HRNet → transformer-based methods
- 重点提一下 WHU 数据集的论文 [1]，说明这是该领域最广泛使用的基准之一

**引用**：[1][3][4][5]

### 2.2 The Segment Anything Model Family

**写什么**：

- SAM (v1) [6]：architecture (ViT encoder + prompt encoder + mask decoder), SA-1B dataset, zero-shot capability
- SAM 2 [7]：key improvements (Hiera encoder, memory attention for video, 6× faster, more accurate on images), SA-V dataset, SAM 2.1 improvements
- 简述 SAM 2 的四个模型大小

**引用**：[6][7]

### 2.3 SAM for Remote Sensing Applications

**写什么**：

- Osco et al. [9] 评估 SAM 在遥感多任务上的表现
- Ren et al. [10] (WACV 2024) "Segment Anything from Space?"——第一个在 overhead imagery 上全面评估 SAM 的工作
- Chen et al. [11] SAM-Adapter：通过 adapter 微调 SAM 适配遥感
- 指出：这些工作主要针对 SAM v1，SAM 2 在遥感建筑物分割上的系统评估仍然缺乏

**引用**：[9][10][11]

---

## 3. Methodology (约1200 words)

### 3.1 Overview

**写什么**：画一个方法流程图（Figure 1），概述整个实验框架。

**Figure 1**：方法流程图，展示——
```
WHU Dataset → Stratified Subset Selection → Three Experiment Groups
                                             ├─ Exp1: Model Size Comparison
                                             ├─ Exp2: Building Type Analysis
                                             └─ Exp3: Prompt Strategy Comparison
```

### 3.2 SAM 2 Architecture Overview

**写什么**：用 1–2 段简要介绍 SAM 2 架构（Hiera encoder, prompt encoder, mask decoder, memory mechanism），说明在处理单张图像时 memory 为空。不需要太深入，审稿人只需要知道你理解这个模型。引用 [7]。

### 3.3 Experiment Design

#### 3.3.1 Experiment 1: Model Size Comparison

**写什么**：

- 四个模型变体：Tiny (38.9M), Small (46M), Base+ (80.8M), Large (224.4M)
- 使用 automatic mask generation mode
- Oracle filtering strategy：对 SAM 2 输出的所有掩码，保留与真值建筑区域重叠率 > 30% 的掩码，合并为最终预测
- 说明 oracle filtering 提供的是 performance upper bound，论文中需要明确说明这一点

#### 3.3.2 Experiment 2: Building Morphology Analysis

**写什么**：

- 五种建筑类型的定义和挑选标准（dense, sparse, large, small, irregular）
- 每组 15–20 张图
- 使用 Exp1 中表现最优的模型配置
- 不仅报告定量指标，还进行 failure case analysis

#### 3.3.3 Experiment 3: Prompt Strategy Comparison

**写什么**：

- 三种提示策略：single-point (centroid), multi-point (5 random interior points), bounding box
- Oracle prompting：所有提示坐标从真值 mask 通过 connected component analysis 自动生成
- 说明 oracle prompting 模拟的是"已知建筑物位置"的理想场景
- 每种策略使用相同的模型（base+），在全部 100 张图上评测

### 3.4 Evaluation Metrics

**写什么**：定义四个评测指标——

- IoU (Intersection over Union)：$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$
- Precision：$\text{Precision} = \frac{|P \cap G|}{|P|}$
- Recall：$\text{Recall} = \frac{|P \cap G|}{|G|}$
- F1 Score：$F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

其中 $P$ 为预测建筑区域，$G$ 为真值建筑区域。

---

## 4. Experiments and Results (约2000 words，论文重头)

### 4.1 Dataset and Implementation Details

**写什么**：

- WHU Building Dataset 介绍：来源（新西兰 Christchurch 航拍）、分辨率（0.3m resampled, 512×512 tiles）、数据量
- 你的子集构造方式：从 test set 中分层抽样 100 张图，覆盖 5 种建筑形态
- 硬件环境：Google Colab, NVIDIA Tesla T4 GPU (16GB)
- SAM 2.1 checkpoints, points_per_side=32, pred_iou_thresh=0.86 等超参

**引用**：[1]

### 4.2 Experiment 1: Effect of Model Size

**放什么**：

- **Table 1**：四个模型的 IoU / F1 / Precision / Recall / Inference Time / GPU Memory
- **分析**（3–4句）：哪个模型最好、tiny 掉了多少精度但快了多少倍、large 相比 base+ 提升是否值得、推荐实际部署用哪个
- **Figure 2**：Accuracy vs. Speed 散点图（横轴=推理时间 ms，纵轴=IoU，4个点标模型名）

### 4.3 Experiment 2: Effect of Building Morphology

**放什么**：

- **Table 2**：五种建筑类型的 IoU / F1 / Precision / Recall
- **分析**（5–6句）：哪种类型最好（预计是 large regular buildings）、哪种最差（预计是 small 或 dense）、原因分析
- **Figure 3**：失败案例可视化——每种类型挑 1 张 IoU 最低的图，展示"原图 / 真值 / 预测 / 错误叠加"四列
- **Figure 4**：成功案例可视化——同样格式，每种类型 IoU 最高的图，与失败案例形成对比
- **失败原因讨论**（重要！）：dense → 相邻建筑粘连; small → 漏检; irregular → 形状不完整; 阴影干扰等

### 4.4 Experiment 3: Effect of Prompt Strategy

**放什么**：

- **Table 3**：三种提示方式的 IoU / F1 / Precision / Recall
- **分析**（3–4句）：box prompting 通常最好（因为提供了明确的空间范围）、single point 最差但仍有一定精度、multi-point 介于两者之间
- **Figure 5**：同一张图三种提示的对比——展示"原图+提示位置标注 / 真值 / 单点预测 / 多点预测 / 框预测"
- **Figure 6**（可选）：提示位置可视化——红点=质心、蓝点=多点、绿框=外接框
- **Table 4**：交叉分析表——建筑类型 × 提示方式的 IoU 矩阵

### 4.5 Summary of Key Findings

**写什么**：用 3–4 句话总结三个实验的核心发现，为 Discussion 做铺垫。

---

## 5. Discussion (约600 words)

### 5.1 Strengths of SAM 2 for Building Segmentation

**写什么**：

- Zero-shot 能力：无需任何遥感数据微调即可达到 X.XX IoU
- Box prompting 表现接近专门训练的模型（如果你的数据支持这个结论的话）
- 多尺度建筑的处理能力（Hiera 分层编码器的贡献）

### 5.2 Limitations and Failure Analysis

**写什么**：

- **Domain gap**：SAM 2 在自然场景上训练，遥感航拍的鸟瞰视角、建筑纹理、阴影模式与训练数据差异大
- **Class-agnostic limitation**：自动模式无法区分建筑和非建筑，实际部署需要额外分类器
- **Dense small buildings**：密集小建筑容易粘连或漏检
- **Oracle prompting 的局限**：本文使用真值生成提示，实际应用中提示质量会更低

### 5.3 Practical Recommendations

**写什么**：给出工程实践建议——

- 模型选择：base+ 是精度与速度的最佳平衡点
- 提示策略：如果有检测框可用（如 YOLO 预检测），优先用 box prompting
- 适用场景：SAM 2 更适合作为半自动标注工具而非全自动分割器

---

## 6. Conclusion (约300 words)

**写什么**：

- **总结**（2–3句）：重述研究目的和主要发现
- **核心贡献**（2–3句）：三组实验的关键结论（最优模型、最优提示、最难场景）
- **未来工作**（2句）：
  - 用 SAM 2 生成伪标签 + 训练轻量分割模型（U-Net）的半监督框架
  - 在更多遥感数据集上验证泛化性（如 Inria Aerial Image Dataset, Massachusetts Building Dataset）
  - 探索 SAM 2 的 adapter/LoRA 微调以缩小域间隙

---

## References (推荐 15–20 篇)

### 核心引用（必须引）

```
[1] Ji, S., Wei, S., & Lu, M. (2018). Fully convolutional networks for
    multi-source building extraction from an open aerial and satellite
    imagery dataset. IEEE Transactions on Geoscience and Remote Sensing,
    57(1), 574–586.
    → WHU 数据集原始论文

[2] Maggiori, E., Tarabalka, Y., Charpiat, G., & Alquier, P. (2017).
    Can semantic labeling methods generalize to any city? The Inria aerial
    image labeling benchmark. IEEE International Geoscience and Remote
    Sensing Symposium (IGARSS).
    → Inria 数据集，在 future work 中提到

[3] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional
    networks for biomedical image segmentation. MICCAI.
    → U-Net，对比方法

[4] Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018).
    Encoder-decoder with atrous separable convolution for semantic image
    segmentation. ECCV.
    → DeepLabV3+，对比方法

[5] Wang, J., et al. (2020). Deep high-resolution representation learning
    for visual recognition. IEEE TPAMI.
    → HRNet，related work 中引用

[6] Kirillov, A., Mintun, E., Ravi, N., et al. (2023). Segment Anything.
    ICCV.
    → SAM v1 原始论文（必引）

[7] Ravi, N., Gabeur, V., Hu, Y.-T., et al. (2024). SAM 2: Segment
    Anything in Images and Videos. arXiv:2408.00714.
    → SAM 2 原始论文（必引）
```

### 重要参考（建议引）

```
[8] Bommasani, R., et al. (2021). On the opportunities and risks of
    foundation models. arXiv:2108.07258.
    → 基础模型综述，Introduction 中引用

[9] Osco, L.P., Wu, Q., et al. (2023). The Segment Anything Model (SAM)
    for remote sensing applications: From zero to one shot.
    International Journal of Applied Earth Observation and Geoinformation.
    → SAM 在遥感中的评估（必引）

[10] Ren, S., Luzi, F., et al. (2024). Segment Anything, from Space?
     WACV 2024.
     → SAM 在 overhead imagery 的评估（强烈建议引）

[11] Chen, T., Zhu, L., et al. (2023). SAM-Adapter: Adapting Segment
     Anything in underperformed scenes. ICCV Workshop.
     → SAM adapter 微调（related work 引用）

[12] Zhang, C., et al. (2023). A comprehensive survey on segment anything
     model for vision and beyond. arXiv:2305.08196.
     → SAM 综述

[13] Li, B., et al. (2023). Semantic-SAM: Segment and recognize anything
     at any granularity. ECCV.
     → SAM 扩展工作

[14] Dosovitskiy, A., et al. (2020). An image is worth 16x16 words:
     Transformers for image recognition at scale. ICLR.
     → ViT，architecture background

[15] Ryali, C., et al. (2023). Hiera: A hierarchical vision transformer
     without the bells-and-whistles. ICML.
     → Hiera 编码器，SAM 2 的骨干网络
```

---

## Figures & Tables 清单

| 编号 | 类型 | 内容 | 来源 |
|------|------|------|------|
| Figure 1 | 流程图 | 方法总体框架 | 自己画 |
| Figure 2 | 散点图 | IoU vs. Inference Speed（4个模型） | Exp 1 |
| Figure 3 | 可视化 | 各类型建筑物分割失败案例 | Exp 2 |
| Figure 4 | 可视化 | 各类型建筑物分割成功案例 | Exp 2 |
| Figure 5 | 可视化 | 三种提示策略对比（含提示位置标注） | Exp 3 |
| Table 1 | 数据表 | 四种模型大小对比指标 | Exp 1 |
| Table 2 | 数据表 | 五种建筑类型分层指标 | Exp 2 |
| Table 3 | 数据表 | 三种提示策略对比指标 | Exp 3 |
| Table 4 | 数据表 | 建筑类型 × 提示方式交叉 IoU | Exp 3 |

---
