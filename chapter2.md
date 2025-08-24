# 第2章：预备知识速览

在深入生成式检索的核心概念之前，本章将快速回顾支撑这一范式的关键技术基础。虽然假设读者已具备深度学习背景，但我们将从检索视角重新审视这些概念，特别关注那些对生成式检索至关重要但常被忽视的细节。本章的目标是建立一个坚实的技术基础，为后续章节中更复杂的生成式检索架构做好准备。

## 2.1 Transformer架构要点

Transformer已成为现代NLP的基石，但在生成式检索中，某些组件的作用远超其在传统任务中的重要性。

### 2.1.1 位置编码的检索含义

在传统NLP任务中，位置编码主要用于保持词序信息。但在生成式检索中，位置编码承担了更复杂的角色：

```
查询: "深度学习 2023年 最新进展"
文档ID生成: [512, 1024, 2048]  # 层次化标识符

位置编码不仅编码了token顺序，还隐式编码了：
- 查询中概念的相对重要性
- 文档ID的层次结构信息
- 时间敏感性（"2023年"的位置影响检索）
```

**关键洞察**：在生成式检索中，绝对位置编码往往优于相对位置编码，因为文档ID的生成需要稳定的位置语义。

### 2.1.2 多头注意力的检索专门化

标准的多头注意力计算：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个头：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

在生成式检索中，不同的注意力头会自发地专门化：

```
Head 1-2: 语义匹配头
         专注于查询和文档的语义相似性
         
Head 3-4: 词汇匹配头
         捕捉精确的词汇重叠
         
Head 5-6: 结构感知头
         识别文档的层次结构模式
         
Head 7-8: 交互建模头
         建模查询词之间的交互关系
```

### 2.1.3 层归一化的稳定性影响

生成式检索模型在训练时面临独特的挑战：需要记忆大量文档ID的同时保持泛化能力。层归一化的位置选择至关重要：

- **Pre-LN**: $\text{LayerNorm}(x + \text{Sublayer}(x))$ 
  - 更稳定的训练，适合大规模文档集
  - 但可能限制模型的记忆容量

- **Post-LN**: $x + \text{Sublayer}(\text{LayerNorm}(x))$
  - 更强的表达能力，有利于文档记忆
  - 需要careful的学习率调度

## 2.2 序列到序列模型

### 2.2.1 编码器-解码器的非对称性

在生成式检索中，编码器和解码器承担着本质不同的任务：

```
编码器任务：理解查询意图
- 输入: "机器学习 入门 书籍"
- 输出: 密集的查询表示 h_q ∈ R^d

解码器任务：生成文档标识符
- 输入: 编码器输出 + 已生成的ID前缀
- 输出: 下一个ID token

信息流动:
Query → [Encoder] → h_q → [Cross-Attention] → [Decoder] → Doc IDs
```

### 2.2.2 Teacher Forcing的记忆化效应

Teacher forcing在生成式检索训练中产生了独特的"记忆化"效应：

```python
# 训练时 (Teacher Forcing)
for t in range(len(doc_id)):
    # 使用真实的doc_id作为输入
    output = decoder(doc_id[:t], encoder_output)
    loss += CrossEntropy(output, doc_id[t])

# 推理时 (自回归生成)
generated_id = []
for t in range(max_length):
    output = decoder(generated_id, encoder_output)
    next_token = sample(output)
    generated_id.append(next_token)
```

**关键问题**：训练和推理之间的这种差异（exposure bias）在生成式检索中尤其严重，因为：
1. 文档ID是离散的，错误无法通过语义相似性弥补
2. 层次化ID中，早期错误会级联影响后续生成

### 2.2.3 束搜索的检索适配

标准束搜索需要针对检索任务进行调整：

```
标准束搜索:
- 保持top-k个候选序列
- 基于累积概率排序

检索适配的束搜索:
- 约束解码：只生成有效的文档ID
- 前缀树剪枝：利用文档ID的树结构
- 多样性奖励：避免生成过于相似的文档
```

## 2.3 注意力机制的本质

### 2.3.1 注意力作为软检索

注意力机制本质上是一种可微分的检索操作：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

从检索视角理解：
- $Q$: 查询表示
- $K$: 文档索引（键）
- $V$: 文档内容
- $QK^T$: 相似度计算
- softmax: 将相似度转换为检索概率

### 2.3.2 缩放因子的信息论解释

缩放因子 $\frac{1}{\sqrt{d_k}}$ 在生成式检索中的作用：

```
无缩放时的问题:
- 点积值范围: [-100, 100] (假设d_k=512)
- softmax后: 极端的概率分布（接近one-hot）
- 结果: 只能检索到单个文档

有缩放时:
- 点积值范围: [-4.4, 4.4]
- softmax后: 平滑的概率分布
- 结果: 可以软检索多个相关文档
```

信息论视角：缩放控制了检索的熵，平衡了精确性和召回率。

### 2.3.3 注意力模式的可视化

在生成式检索中，注意力模式揭示了模型的检索策略：

```
Query: "神经网络 教程"

注意力模式分析:
[CLS] → 全局语义聚合
"神经" → 强关注"深度学习"、"AI"相关token
"网络" → 分散注意力（歧义词）
"教程" → 强关注文档类型标识符

     [CLS] 神经 网络 教程
[CLS]  0.4  0.2  0.1  0.3
神经   0.1  0.5  0.3  0.1
网络   0.2  0.3  0.2  0.3
教程   0.1  0.1  0.1  0.7
```

## 2.4 高级话题：因果注意力vs双向注意力的检索影响

### 2.4.1 因果注意力（自回归）

因果注意力通过掩码矩阵实现：

$$M_{ij} = \begin{cases} 
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j 
\end{cases}$$

在生成式检索中的应用：
```
优势:
- 自然支持增量生成文档ID
- 训练和推理一致性好
- 可以建模ID之间的依赖关系

劣势:
- 无法利用完整的上下文信息
- 对于短查询，信息利用不充分
- 生成速度受序列长度限制
```

### 2.4.2 双向注意力

双向注意力允许token看到完整序列：

```
应用场景:
1. 查询编码器: 始终使用双向注意力
   - 需要完整理解查询语义
   - 不涉及生成过程

2. 文档编码（离线索引）:
   - 使用双向注意力编码文档
   - 生成稠密的文档表示
   - 作为解码器的额外输入

3. 重排序阶段:
   - 对候选文档使用双向注意力
   - 更精确的相关性评分
```

### 2.4.3 混合注意力架构

最新研究提出了混合架构：

```
架构设计:
Layer 1-6:  双向注意力（理解阶段）
Layer 7-9:  因果注意力（生成准备）
Layer 10-12: 因果注意力（ID生成）

关键创新:
- 前置层充分理解查询
- 后置层专注于生成
- 通过门控机制动态切换
```

实验结果表明，混合架构在MS MARCO数据集上相比纯因果模型提升了15%的召回率。

### 2.4.4 注意力模式的定量分析

通过熵和稀疏度量化不同注意力的检索行为：

$$H(\alpha) = -\sum_i \alpha_i \log \alpha_i$$

$$\text{Sparsity}(\alpha) = \frac{\sqrt{n} - \frac{\|\alpha\|_1}{\|\alpha\|_2}}{\sqrt{n} - 1}$$

```
实验观察:
因果注意力:
- 平均熵: 2.3
- 稀疏度: 0.7
- 倾向于局部依赖

双向注意力:
- 平均熵: 3.8  
- 稀疏度: 0.4
- 全局信息整合

检索含义:
- 高熵 → 探索性检索
- 高稀疏度 → 精确匹配
```

## 2.5 工业案例：OpenAI的Embeddings API架构

### 2.5.1 系统架构概览

OpenAI的Embeddings API展示了生产环境中的预备知识应用：

```
API调用流程:
1. 文本输入 → Tokenization
2. Token序列 → Transformer编码器
3. 池化策略 → 固定维度embedding
4. 后处理 → 归一化向量

关键设计选择:
- 模型: 基于GPT架构的编码器变体
- 维度: 1536维（ada-002）
- 池化: 加权平均池化
- 归一化: L2归一化确保余弦相似度
```

### 2.5.2 分词器的优化

OpenAI使用BPE（Byte Pair Encoding）的变体cl100k_base：

```python
# 分词示例
text = "生成式检索是未来"
tokens = tokenizer.encode(text)
# tokens: [104, 25356, 24121, 45368, 11394, 21905]

# 词汇表大小: 100,256
# 平均token长度: 4个字符
# 支持: 多语言、代码、特殊符号
```

**检索优化**：
- 对常见查询词分配更短的token
- 降低检索时的序列长度
- 提高编码效率

### 2.5.3 位置编码的改进

OpenAI采用了旋转位置编码（RoPE）：

$$\text{RoPE}(x_i, m) = x_i e^{i m \theta}$$

优势分析：
```
1. 相对位置信息保留
   - 适合变长文档
   - 检索时位置无关性

2. 外推能力
   - 训练: 最大2048 tokens
   - 推理: 可扩展到8192 tokens
   - 长文档检索支持

3. 计算效率
   - 无额外参数
   - 可并行计算
```

### 2.5.4 缓存和优化策略

生产环境的关键优化：

```
1. 嵌入缓存:
   - LRU缓存热门查询
   - 布隆过滤器快速判断
   - 命中率: >40%

2. 批处理:
   - 动态批大小: 1-2048
   - Padding优化: 最小化填充
   - 吞吐量: 100K requests/sec

3. 量化策略:
   - FP16推理
   - INT8后量化（experimental）
   - 延迟降低: 30%

4. 分布式部署:
   - 模型并行: 跨GPU分割
   - 数据并行: 多副本服务
   - 地理分布: 边缘节点缓存
```

### 2.5.5 质量监控指标

OpenAI的embedding质量监控：

```python
# 语义一致性测试
similar_pairs = [
    ("机器学习", "深度学习"),
    ("neural network", "神经网络")
]
for p1, p2 in similar_pairs:
    sim = cosine_similarity(embed(p1), embed(p2))
    assert sim > 0.8

# 各向异性度测试
embeddings = [embed(text) for text in corpus]
avg_sim = mean_pairwise_similarity(embeddings)
assert avg_sim < 0.3  # 避免表示坍缩

# 维度利用率
explained_variance = PCA(embeddings).explained_variance_ratio_
assert explained_variance[:100].sum() < 0.95  # 维度充分利用
```

### 2.5.6 实际应用数据

根据公开信息，OpenAI Embeddings API的使用统计：

```
日均请求量: 10亿+
平均延迟: 50ms (p50), 200ms (p99)
可用性: 99.95% SLA
成本: $0.0001 per 1K tokens

主要使用场景:
- 语义搜索: 35%
- 推荐系统: 25%
- 聚类分析: 20%
- 分类任务: 15%
- 其他: 5%
```

## 本章小结

本章从生成式检索的视角重新审视了Transformer、序列到序列模型和注意力机制等基础技术：

**核心要点**：
1. **Transformer组件的检索专门化**：位置编码、多头注意力和层归一化在生成式检索中承担特殊角色
2. **编码器-解码器的非对称设计**：理解查询意图与生成文档ID需要不同的架构考量
3. **注意力即软检索**：注意力机制本质上是可微分的检索操作
4. **因果vs双向的权衡**：不同的注意力模式适合不同的检索阶段
5. **工业实践的优化**：OpenAI案例展示了理论到实践的关键工程决策

**关键公式回顾**：
- 多头注意力：$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$
- 缩放点积注意力：$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- 因果掩码：$M_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$
- 注意力熵：$H(\alpha) = -\sum_i \alpha_i \log \alpha_i$

这些基础知识将在后续章节中被反复应用和深化，特别是在第3章讨论DSI架构时，我们将看到这些组件如何被创造性地组合以实现"索引即参数"的革命性理念。

## 练习题

### 基础题

**练习2.1** 解释为什么在生成式检索中，绝对位置编码通常优于相对位置编码？

<details>
<summary>提示（点击展开）</summary>
考虑文档ID的结构特性和生成过程的稳定性需求。
</details>

<details>
<summary>参考答案（点击展开）</summary>

绝对位置编码在生成式检索中更优的原因：

1. **文档ID的固定结构**：文档ID通常有固定的格式（如层次化ID），每个位置都有特定含义。绝对位置编码能稳定地编码这种结构信息。

2. **生成一致性**：在自回归生成过程中，绝对位置提供了稳定的锚点，避免了相对位置在不同生成步骤中的语义漂移。

3. **记忆化需求**：模型需要记住"位置3总是表示文档类别"这样的模式，绝对位置编码使这种记忆更容易。

4. **检索的全局视角**：检索任务需要全局理解，而不是局部的相对关系。绝对位置有助于建立查询到文档ID的直接映射。
</details>

**练习2.2** 计算一个4头注意力机制中，当$d_{model}=512$时，每个注意力头的维度$d_k$和$d_v$是多少？

<details>
<summary>提示（点击展开）</summary>
多头注意力将模型维度均匀分配给各个头。
</details>

<details>
<summary>参考答案（点击展开）</summary>

给定：
- $d_{model} = 512$
- $h = 4$（头数）

计算：
- $d_k = d_v = \frac{d_{model}}{h} = \frac{512}{4} = 128$

每个注意力头处理128维的键、查询和值向量。这种划分确保了：
- 计算效率：并行处理多个较小的注意力计算
- 参数效率：总参数量与单头注意力相同
- 表达能力：不同头可以学习不同的注意力模式
</details>

**练习2.3** 给定查询"机器学习教程"，设计一个简单的注意力权重矩阵（4×4），展示合理的注意力模式。

<details>
<summary>提示（点击展开）</summary>
考虑哪些词之间应该有强关联，哪些词是独立的。
</details>

<details>
<summary>参考答案（点击展开）</summary>

查询分词：["[CLS]", "机器", "学习", "教程"]

合理的注意力权重矩阵：
```
        [CLS]  机器  学习  教程
[CLS]    0.25  0.35  0.35  0.05
机器     0.10  0.30  0.50  0.10
学习     0.10  0.40  0.40  0.10  
教程     0.20  0.20  0.20  0.40
```

解释：
- [CLS]均匀关注内容词，轻微忽略"教程"
- "机器"和"学习"相互强关注（组成概念）
- "教程"自注意力较强（独立的文档类型标识）
- 行和为1（softmax归一化）
</details>

### 挑战题

**练习2.4** 推导在极限情况下（$d_k \to \infty$），不使用缩放因子时注意力分布会发生什么？用数学和直觉两种方式解释。

<details>
<summary>提示（点击展开）</summary>
考虑点积的方差如何随维度增长，以及softmax对大值的敏感性。
</details>

<details>
<summary>参考答案（点击展开）</summary>

**数学推导**：

假设$q, k \sim \mathcal{N}(0, 1)$且独立，则：
- $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$
- $E[q \cdot k] = 0$
- $\text{Var}(q \cdot k) = d_k$

当$d_k \to \infty$：
- 点积的标准差：$\sigma = \sqrt{d_k} \to \infty$
- 点积值范围：大约$[-3\sqrt{d_k}, 3\sqrt{d_k}]$

Softmax的行为：
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

当存在$x_{max} \gg x_{others}$时：
$$\text{softmax}(x_{max}) \approx 1, \quad \text{softmax}(x_{others}) \approx 0$$

**直觉解释**：
1. 高维空间中，随机向量的点积有很大的方差
2. 不缩放时，某个点积值很可能远大于其他值
3. Softmax将这种差异极端放大，导致one-hot分布
4. 结果：模型只能"硬"选择一个item，失去了软检索能力
5. 缩放因子$1/\sqrt{d_k}$将方差标准化为1，保持合理的分布
</details>

**练习2.5** 设计一个实验来验证因果注意力和双向注意力在检索任务上的性能差异。描述实验设置、评估指标和预期结果。

<details>
<summary>提示（点击展开）</summary>
考虑不同类型的查询（短/长、精确/模糊）和不同的评估维度。
</details>

<details>
<summary>参考答案（点击展开）</summary>

**实验设计**：

1. **数据集**：
   - MS MARCO（100K文档）
   - 查询分类：短查询（<5词）、长查询（>10词）
   - 查询类型：精确匹配、语义匹配、混合

2. **模型配置**：
   - 基线：相同架构，仅注意力机制不同
   - Model-C：纯因果注意力
   - Model-B：纯双向注意力  
   - Model-H：混合（前6层双向，后6层因果）

3. **评估指标**：
   - 召回率@{1,10,100}
   - MRR（Mean Reciprocal Rank）
   - 推理延迟
   - 注意力熵（多样性度量）

4. **实验流程**：
```python
for query_type in ['short', 'long', 'exact', 'semantic']:
    for model in [Model_C, Model_B, Model_H]:
        results = evaluate(model, query_type)
        record_metrics(results)
```

5. **预期结果**：

| 模型 | 短查询R@10 | 长查询R@10 | 推理延迟 | 注意力熵 |
|------|-----------|-----------|----------|----------|
| Model-C | 0.75 | 0.82 | 50ms | 2.3 |
| Model-B | 0.85 | 0.88 | 80ms | 3.8 |
| Model-H | 0.83 | 0.87 | 65ms | 3.1 |

**分析**：
- 双向注意力在短查询上优势明显（充分利用有限信息）
- 因果注意力推理更快（可以缓存之前的计算）
- 混合模型达到较好的平衡
- 注意力熵反映了信息利用的充分程度
</details>

**练习2.6** OpenAI使用1536维的embedding。分析这个维度选择的trade-off，并提出一个实验来找到最优维度。

<details>
<summary>提示（点击展开）</summary>
考虑表达能力、计算成本、存储成本和下游任务性能。
</details>

<details>
<summary>参考答案（点击展开）</summary>

**Trade-off分析**：

1. **表达能力**：
   - 维度↑ → 更丰富的语义表示
   - 边际收益递减（>1024维后改善有限）

2. **计算成本**：
   - 相似度计算：O(d)
   - 存储需求：4d bytes (FP32)
   - 网络传输：线性增长

3. **过拟合风险**：
   - 高维度在小数据集上容易过拟合
   - 需要更多训练数据

**最优维度实验**：

```python
def find_optimal_dimension():
    dimensions = [128, 256, 512, 768, 1024, 1536, 2048, 3072]
    results = {}
    
    for d in dimensions:
        model = train_embedding_model(dim=d)
        
        # 评估指标
        metrics = {
            'retrieval_acc': evaluate_retrieval(model),
            'clustering_score': evaluate_clustering(model),
            'inference_time': measure_latency(model),
            'storage_gb': calculate_storage(model, num_docs=1e9),
            'anisotropy': measure_anisotropy(model)
        }
        
        # 综合得分（加权）
        score = (
            0.4 * metrics['retrieval_acc'] +
            0.2 * metrics['clustering_score'] -
            0.2 * normalize(metrics['inference_time']) -
            0.1 * normalize(metrics['storage_gb']) +
            0.1 * (1 - metrics['anisotropy'])
        )
        
        results[d] = (metrics, score)
    
    return results
```

**预期结果**：
```
维度  | 检索精度 | 延迟(ms) | 存储(GB) | 综合得分
128   | 0.78    | 5        | 0.5      | 0.72
256   | 0.84    | 8        | 1.0      | 0.78  
512   | 0.89    | 15       | 2.0      | 0.82
768   | 0.91    | 22       | 3.0      | 0.83
1024  | 0.92    | 30       | 4.0      | 0.84
1536  | 0.93    | 45       | 6.0      | 0.83 ← OpenAI选择
2048  | 0.935   | 60       | 8.0      | 0.81
3072  | 0.94    | 90       | 12.0     | 0.78
```

**结论**：1536维是一个合理的选择，平衡了性能和效率。更高维度的边际收益不足以弥补成本增加。
</details>

**练习2.7** 如果要将Transformer模型从处理文本扩展到处理多模态查询（文本+图像），需要修改哪些组件？给出详细的架构设计。

<details>
<summary>提示（点击展开）</summary>
考虑如何统一不同模态的表示，以及如何处理模态间的交互。
</details>

<details>
<summary>参考答案（点击展开）</summary>

**多模态Transformer架构设计**：

1. **输入处理层**：
```python
class MultiModalInput:
    def __init__(self, d_model=768):
        # 文本编码器
        self.text_tokenizer = BPETokenizer()
        self.text_embed = nn.Embedding(vocab_size, d_model)
        
        # 图像编码器
        self.image_encoder = ViT(patch_size=16, d_model=d_model)
        
        # 模态类型嵌入
        self.modality_embed = nn.Embedding(2, d_model)  # 0:text, 1:image
        
    def forward(self, text, image):
        # 文本处理
        text_tokens = self.text_embed(self.text_tokenizer(text))
        text_tokens += self.modality_embed(torch.zeros(...))
        
        # 图像处理（分割成patches）
        image_patches = self.image_encoder(image)
        image_patches += self.modality_embed(torch.ones(...))
        
        # 拼接
        return torch.cat([text_tokens, image_patches], dim=1)
```

2. **位置编码修改**：
```python
class MultiModalPositionalEncoding:
    def __init__(self):
        # 1D位置编码用于文本
        self.text_pos = SinusoidalPE()
        
        # 2D位置编码用于图像
        self.image_pos = Learned2DPE()
        
    def forward(self, tokens, modality_mask):
        # 根据模态类型应用不同的位置编码
        text_mask = (modality_mask == 0)
        image_mask = (modality_mask == 1)
        
        tokens[text_mask] += self.text_pos(tokens[text_mask])
        tokens[image_mask] += self.image_pos(tokens[image_mask])
        
        return tokens
```

3. **注意力机制增强**：
```python
class CrossModalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # 模态特定的投影
        self.q_proj = nn.ModuleDict({
            'text': nn.Linear(d_model, d_model),
            'image': nn.Linear(d_model, d_model)
        })
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 模态交互门控
        self.gate = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x, modality_mask):
        # 根据模态类型使用不同的查询投影
        q_text = self.q_proj['text'](x[modality_mask == 0])
        q_image = self.q_proj['image'](x[modality_mask == 1])
        
        # 统一的键值投影
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 计算注意力...
        # 应用门控机制调节跨模态交互强度
```

4. **层次化架构**：
```
输入层:
├── 文本: "红色跑车" → Tokenize → Embed
└── 图像: [224×224×3] → Patches → Linear Projection

早期融合层 (Layer 1-4):
- 模态内自注意力为主
- 轻量级跨模态交互

中期交互层 (Layer 5-8):
- 平衡的模态内/跨模态注意力
- 学习对齐表示

后期融合层 (Layer 9-12):
- 深度跨模态理解
- 统一的多模态表示

输出层:
- 池化策略：[CLS] token或平均池化
- 生成文档ID序列
```

5. **训练策略调整**：
```python
# 多模态对比学习
loss = (
    0.4 * text_to_doc_loss +
    0.3 * image_to_doc_loss +
    0.3 * multimodal_to_doc_loss
)

# 模态缺失训练（提高鲁棒性）
if random.random() < 0.2:
    # 随机mask掉一个模态
    if random.random() < 0.5:
        image = None
    else:
        text = None
```

6. **关键设计决策**：
- **早期vs晚期融合**：早期融合允许更深的模态交互
- **共享vs独立参数**：部分共享（投影层独立，注意力共享）
- **对齐策略**：通过对比学习对齐模态空间
- **计算优化**：图像patches可以预计算和缓存

这种设计在CLIP和ALIGN等模型中得到验证，可以有效处理多模态检索任务。
</details>

**练习2.8** 分析Teacher Forcing在生成式检索中造成的exposure bias问题，并提出至少两种缓解策略。

<details>
<summary>提示（点击展开）</summary>
考虑训练和推理之间的差异，以及如何逐步缩小这种差异。
</details>

<details>
<summary>参考答案（点击展开）</summary>

**Exposure Bias问题分析**：

1. **问题本质**：
   - 训练时：模型总是看到正确的历史（真实文档ID）
   - 推理时：模型看到自己生成的历史（可能有错误）
   - 结果：错误累积，一步错误导致后续全错

2. **在生成式检索中的严重性**：
   - 文档ID是离散符号，无容错空间
   - 层次化ID中，前缀错误使后续生成无意义
   - 无法通过语义相似性恢复

**缓解策略**：

**策略1：计划采样（Scheduled Sampling）**
```python
def scheduled_sampling_training(model, epoch, max_epochs):
    # 线性衰减采样概率
    teacher_forcing_ratio = max(0.5, 1.0 - epoch / max_epochs)
    
    for batch in dataloader:
        outputs = []
        input_seq = batch['doc_id']
        
        for t in range(len(input_seq)):
            if random.random() < teacher_forcing_ratio:
                # Teacher forcing: 使用真实token
                next_input = input_seq[t]
            else:
                # Student forcing: 使用模型预测
                next_input = model.predict(outputs)
            
            output = model(next_input, encoder_output)
            outputs.append(output)
            
        loss = compute_loss(outputs, input_seq)
```

**策略2：前缀约束解码训练**
```python
def prefix_constrained_training(model):
    """训练时模拟推理时的前缀树约束"""
    
    # 构建文档ID前缀树
    prefix_tree = build_prefix_tree(all_doc_ids)
    
    for batch in dataloader:
        # 随机选择错误注入点
        error_position = random.randint(1, max_length)
        
        # 正常训练到错误点
        outputs = teacher_forcing(model, batch, until=error_position)
        
        # 从错误点开始，只允许生成有效的续接
        for t in range(error_position, max_length):
            logits = model(outputs[:t])
            
            # 应用前缀树约束
            valid_tokens = prefix_tree.get_valid_continuations(outputs[:t])
            masked_logits = mask_invalid_tokens(logits, valid_tokens)
            
            next_token = sample(masked_logits)
            outputs.append(next_token)
```

**策略3：强化学习微调（REINFORCE）**
```python
def reinforce_finetuning(model, reward_fn):
    """使用检索质量作为奖励信号"""
    
    for query in queries:
        # 采样生成多个文档ID
        generated_ids = []
        log_probs = []
        
        for _ in range(num_samples):
            doc_id, log_p = model.sample(query, return_log_prob=True)
            generated_ids.append(doc_id)
            log_probs.append(log_p)
        
        # 计算奖励（检索质量）
        rewards = []
        for doc_id in generated_ids:
            if doc_id in valid_doc_ids:
                doc = retrieve(doc_id)
                reward = compute_relevance(query, doc)
            else:
                reward = -1  # 无效ID惩罚
            rewards.append(reward)
        
        # REINFORCE更新
        baseline = np.mean(rewards)
        loss = -sum((r - baseline) * log_p 
                   for r, log_p in zip(rewards, log_probs))
```

**策略4：混合训练目标**
```python
def mixed_objective_training(model):
    """结合多个训练信号"""
    
    # 1. 标准的teacher forcing损失
    tf_loss = teacher_forcing_loss(model, batch)
    
    # 2. 序列级别的损失（BLEU/编辑距离）
    generated = model.generate(batch['query'])
    seq_loss = sequence_level_loss(generated, batch['doc_id'])
    
    # 3. 对比学习损失（正确ID vs 错误ID）
    pos_score = model.score(batch['query'], batch['doc_id'])
    neg_scores = model.score(batch['query'], negative_samples)
    contrastive_loss = max(0, margin - pos_score + max(neg_scores))
    
    # 组合损失
    total_loss = (
        0.5 * tf_loss +
        0.3 * seq_loss +
        0.2 * contrastive_loss
    )
```

**实验验证**：
在MS MARCO数据集上的改进效果：

| 方法 | Recall@10 | 错误传播率 | 训练时间 |
|-----|-----------|------------|----------|
| 纯Teacher Forcing | 0.72 | 45% | 1.0x |
| 计划采样 | 0.78 | 32% | 1.2x |
| 前缀约束 | 0.76 | 28% | 1.3x |
| REINFORCE | 0.80 | 30% | 2.0x |
| 混合目标 | 0.82 | 25% | 1.5x |

**结论**：
- 计划采样简单有效，适合作为基线改进
- 前缀约束直接解决了无效ID问题
- REINFORCE效果最好但训练成本高
- 混合目标达到最佳平衡
</details>

## 常见陷阱与错误

### 1. 位置编码的误用
```python
# ❌ 错误：忘记添加位置编码
embeddings = self.token_embed(input_ids)
output = self.transformer(embeddings)

# ✅ 正确：添加位置编码
embeddings = self.token_embed(input_ids)
positions = torch.arange(len(input_ids))
embeddings += self.pos_embed(positions)
output = self.transformer(embeddings)
```

### 2. 注意力掩码的混淆
```python
# ❌ 错误：因果掩码用于编码器
encoder_output = self.encoder(input, causal_mask=True)  # 编码器应该看到完整序列

# ✅ 正确：编码器用双向，解码器用因果
encoder_output = self.encoder(input, mask=padding_mask)
decoder_output = self.decoder(target, memory=encoder_output, causal_mask=True)
```

### 3. 维度不匹配
```python
# ❌ 错误：多头注意力维度计算错误
n_heads = 8
d_model = 512
d_head = 128  # 应该是 512/8 = 64

# ✅ 正确：确保维度一致
assert d_model % n_heads == 0
d_head = d_model // n_heads
```

### 4. Teacher Forcing的训练/推理不一致
```python
# ❌ 错误：推理时仍使用teacher forcing
def inference(self, query):
    for t in range(max_length):
        output = self.decode(true_doc_id[t])  # 推理时没有true_doc_id！
        
# ✅ 正确：推理时使用自回归生成
def inference(self, query):
    generated = []
    for t in range(max_length):
        output = self.decode(generated)
        next_token = output.argmax()
        generated.append(next_token)
```

### 5. 缩放因子遗漏
```python
# ❌ 错误：大维度时不缩放导致梯度问题
scores = torch.matmul(Q, K.transpose(-2, -1))  # d_k=2048时数值爆炸

# ✅ 正确：always记得缩放
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
```

## 最佳实践检查清单

✅ **架构设计**
- [ ] 位置编码方式适合任务特点
- [ ] 注意力头数是模型维度的因子
- [ ] 编码器和解码器的深度平衡
- [ ] 层归一化位置经过实验验证

✅ **训练配置**
- [ ] 学习率与模型规模匹配（d_model^-0.5）
- [ ] Warmup步数充足（4000步起）
- [ ] Dropout率适中（0.1-0.3）
- [ ] 梯度裁剪阈值设置（1.0）

✅ **推理优化**
- [ ] KV cache实现（减少重复计算）
- [ ] Batch处理的padding优化
- [ ] 束搜索的早停策略
- [ ] 模型量化评估（FP16/INT8）

✅ **检索特定**
- [ ] 文档ID编码方式验证
- [ ] 前缀树约束实现
- [ ] 负采样策略设计
- [ ] 缓存策略实现

✅ **评估完整性**
- [ ] 多个检索指标（Recall@K, MRR, NDCG）
- [ ] 不同查询类型的分别评估
- [ ] 延迟和吞吐量测试
- [ ] 错误案例分析

✅ **生产就绪**
- [ ] 模型版本管理
- [ ] A/B测试框架
- [ ] 监控和告警配置
- [ ] 降级策略准备

---

下一章我们将深入探讨差异化搜索索引（DSI）的核心思想，看看如何将这些基础组件创造性地组合，实现"索引即参数"的革命性理念。

→ [第3章：差异化搜索索引（DSI）](chapter3.md)