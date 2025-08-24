# 第11章：序列推荐与生成模型

序列推荐是推荐系统中的核心问题之一，它通过分析用户的历史行为序列来预测下一个可能的交互项目。随着Transformer架构在NLP领域的巨大成功，研究者们开始探索如何将这些强大的生成模型应用于序列推荐任务。本章将深入探讨GPT4Rec等生成式序列推荐模型，理解如何将推荐问题转化为序列生成任务，以及如何设计高效的个性化生成策略。我们将特别关注长序列建模的挑战，并通过Amazon的实际案例了解这些技术在工业界的应用。

**学习目标：**
- 掌握GPT4Rec的核心架构和训练机制
- 理解用户行为序列的多种编码方法
- 学会设计个性化的生成策略
- 了解长序列建模的优化技术
- 通过工业案例理解实际部署挑战

## 11.1 GPT4Rec及其变体

### 11.1.1 从SASRec到GPT4Rec的演进

序列推荐模型的发展经历了从RNN到Transformer的重要转变。SASRec (Self-Attentive Sequential Recommendation) 首次将自注意力机制引入序列推荐，证明了Transformer架构在捕捉用户行为模式方面的优越性。

```
传统序列模型演进路径：
RNN/LSTM → GRU4Rec → SASRec → BERT4Rec → GPT4Rec
```

GPT4Rec的核心创新在于将推荐任务完全形式化为自回归生成任务：

$$p(v_{n+1}|v_1, v_2, ..., v_n) = \prod_{i=1}^{|V|} p(t_i|v_1, ..., v_n, t_1, ..., t_{i-1})$$

其中$v_i$表示用户交互的第$i$个物品，$t_i$表示物品ID的第$i$个token。

### 11.1.2 GPT4Rec架构详解

GPT4Rec采用标准的GPT架构，但针对推荐任务进行了关键适配：

```
输入层设计：
[User] [Item_1] [Item_2] ... [Item_n] [MASK]
   ↓       ↓        ↓            ↓        ↓
Embedding Layer (Item + Position + Time)
   ↓       ↓        ↓            ↓        ↓
Multi-Head Self-Attention (Causal Mask)
   ↓       ↓        ↓            ↓        ↓
Feed-Forward Network
   ↓       ↓        ↓            ↓        ↓
Layer Norm + Residual
   ↓
Output: Next Item Probability Distribution
```

关键组件说明：

1. **物品嵌入层**：将物品ID映射到高维向量空间
   $$\mathbf{e}_i = \text{Embed}(v_i) \in \mathbb{R}^d$$

2. **位置编码**：捕捉序列中的顺序信息
   $$\mathbf{p}_i = \text{PE}(i) \in \mathbb{R}^d$$

3. **因果注意力掩码**：确保模型只能看到历史信息
   $$\text{Mask}(i,j) = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

### 11.1.3 主要变体比较

不同的生成式序列推荐模型在架构和训练目标上各有特点：

**BERT4Rec**：采用双向Transformer，通过掩码语言模型(MLM)训练
- 优势：能够利用未来信息进行更准确的预测
- 劣势：推理时需要特殊处理，不适合实时生成

**GPT2Rec**：使用GPT-2的预训练权重进行初始化
- 优势：利用大规模预训练知识
- 劣势：需要额外的领域适配

**P5 (Pretrain, Personalized Prompt, and Predict Paradigm)**：
- 将所有推荐任务统一为文本生成
- 支持多任务学习和零样本泛化

## 11.2 用户行为序列的编码

### 11.2.1 序列表示方法

用户行为序列的表示直接影响模型的学习效果。主要有三种表示策略：

**1. 原子化表示**
每个物品作为独立的token：
```
用户序列: [手机, 耳机, 充电器, 手机壳]
编码: [2451, 1832, 3421, 892]
```

**2. 层次化表示**
将物品分解为类别+属性：
```
用户序列: [电子/手机/iPhone, 配件/音频/AirPods]
编码: [[15, 23, 145], [18, 45, 298]]
```

**3. 语义化表示**
使用预训练的文本编码器：
```
用户序列: ["iPhone 14 Pro", "AirPods Pro"]
编码: [BERT("iPhone 14 Pro"), BERT("AirPods Pro")]
```

### 11.2.2 位置编码策略

标准的正弦位置编码在推荐场景中可能不够灵活，因此出现了多种改进方案：

**相对位置编码**：
$$\text{RPE}(i, j) = \mathbf{w}_{clip(j-i, -K, K)}$$

其中$K$是最大相对距离，$\mathbf{w}$是可学习的参数。

**时间感知位置编码**：
结合实际时间间隔：
$$\mathbf{p}_{i,j} = \mathbf{p}_{pos}(j-i) + \mathbf{p}_{time}(\Delta t_{i,j})$$

### 11.2.3 时间信息的融合

用户行为的时间模式对推荐至关重要。GPT4Rec通过多种方式融合时间信息：

1. **时间间隔嵌入**：
   $$\mathbf{t}_i = \text{TimeEmbed}(\log(1 + \Delta t_i))$$

2. **周期性编码**：
   捕捉日、周、月等周期模式：
   $$\mathbf{c}_i = [\sin(2\pi t_i/T_d), \cos(2\pi t_i/T_d), ...]$$

3. **时间衰减注意力**：
   $$\text{Attention}_{time}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d}} - \lambda \cdot \Delta T)V$$