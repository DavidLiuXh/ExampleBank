本文会先介绍几篇关于 RAG 优化的论文，再记录一些常见的RAG工程实践经验。

在大模型实际落地的时候，存在一些问题，主要集中在以下方面：

+ 缺少垂直领域知识：虽然大模型压缩了大量的人类知识，但在垂直场景上明显存在短板，需要专业化的服务去解决特定问题。
+ 存在幻觉、应用有一定门槛：在大模型使用上有一些幻觉、合规问题，没有办法很好地落地，配套工作不足，缺乏现成的方案来管理非结构化文本、进行测试、运营和管理等。
+ 存在重复建设：各业务孤立摸索，资产无法沉淀，存在低水平重复建设，对公司来说ROI低，不够高效。

 站在应用的角度，需要一种能够有效解决大模型在垂直领域知识短板、降低应用门槛、提高效率并发挥规模优势的技术方案。

当前业内RAG是一种相对有效的解决上面问题的方案(平台化能力、开箱即用垂直私域数据)。

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgzy7yYvYRbYdiaMjowFucib9hQjYoichIbZjzwVl8zDhmw0dHcHKlHyJBw/640?wx_fmt=other&from=appmsg)

实现一个基本的不难，但要真正优化RAG的效果，还需要投入大量的精力。这里根据个人的理解，梳理总结一些常见的RAG优化方法，以便日后在实践RAG时提供参考。

+ 首先，介绍几篇关于RAG优化的论文；
+ 然后，记录一些常见的RAG工程实践经验；

 希望能给大家带来一些参考，后续个人也会重点投入在RAG这块，希望和大家多多交流。 

## RAG介绍
![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIg4FQRudkse3Zg5K3BaNNbvdpdrcUm44UlAoLepH6ibFRoYHonRhRVIwA/640?wx_fmt=other&from=appmsg) 

RAG 是 “Retrieval-Augmented Generation”（检索增强生成）的缩写，它通过结合检索系统和生成模型来提高语言生成的准确性和相关性。

RAG 的优势在于它能够在生成响应时引入外部知识，这使得生成的内容更加准确和信息丰富，对于处理需要专业知识或大量背景信息的问题尤其有效。随着大型语言模型（LLMs）的发展，RAG 技术也在不断进化，以适应更长的上下文和更复杂的查询。

+ 目前，大部分公司倾向于使用 RAG方法进行信息检索，因为相比长文本的使用成本，使用向量数据库的成本更低。
+ 而在 RAG 应用过程中，一些公司会使用微调的 Embedding Model，以增强RAG 的检索能力；另一些些公司会选择使用知识图谱或者ES 等非向量数据库的 RAG 方法。
+ 大多数第三方个人和企业开发者会使用集成好的 RAG 框架（例如llamaindex、langchain、etcs），或者直接使用LLMOps 里内建的RAG 工具。

## 相关优化论文
### RAPTOR
论文地址：[https://arxiv.org/pdf/2401.18059](https://arxiv.org/pdf/2401.18059) 

传统的RAG方法通常仅检索较短的连续文本块，这限制了对整体文档上下文的全面理解。RAPTOR(Recursive Abstractive Processing for Tree-Organized Retrieval)通过递归嵌入、聚类和总结文本块，构建一个自底向上的树形结构，在推理时从这棵树中检索信息，从而在不同抽象层次上整合来自长文档的信息。

1.**树形结构构建：**

+ **文本分块：**首先将检索语料库分割成短的、连续的文本块。
+ **嵌入和聚类：**使用SBERT（基于BERT的编码器）将这些文本块嵌入，然后使用高斯混合模型（GMM）进行聚类。
+ **摘要生成：**对聚类后的文本块使用语言模型生成摘要，这些摘要文本再重新嵌入，并继续聚类和生成摘要，直到无法进一步聚类，最终构建出多层次的树形结构。

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgvibm3SskO173K5O3iaBhMPU5dE0c00Q7icLb6F2NZQuWDPDtbMN8n6czA/640?wx_fmt=other&from=appmsg)![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIg9pn7cjXgtDXBuOiaQWvSSnwndx3hR55SjpqO0nlWTicVEWOmfjfG8u9Q/640?wx_fmt=other&from=appmsg)  

2.**查询方法：**

+ **树遍历：**从树的根层开始，逐层选择与查询向量余弦相似度最高的节点，直到到达叶节点，将所有选中的节点文本拼接形成检索上下文。
+ **平铺遍历：**将整个树结构平铺成一个单层，将所有节点同时进行比较，选出与查询向量余弦相似度最高的节点，直到达到预定义的最大token数。

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgl7THoVlqh4uEUc3YiahSnWiajGCY7EVGOuqvernd6qa9JROn8ck5aCVA/640?wx_fmt=other&from=appmsg)  

3.**实验结果：**RAPTOR在多个任务上显著优于传统的检索增强方法，特别是在涉及复杂多步推理的问答任务中。RAPTOR与GPT-4结合后，在QuALITY基准上的准确率提高了20%。

+ 代码：RAPTOR的源代码将在GitHub上公开。
+ 数据集：实验中使用了NarrativeQA、QASPER和QuALITY等问答数据集。

参考视频：[https://www.youtube.com/watch?v=jbGchdTL7d0](https://www.youtube.com/watch?v=jbGchdTL7d0) 

### Self-RAG
论文地址：[https://arxiv.org/pdf/2310.11511](https://arxiv.org/pdf/2310.11511) 

SELF-RAG（自反思检索增强生成）是一种新框架，通过让语言模型在生成过程中进行自我反思，来提高生成质量和事实正确性，同时不损失语言模型的多样性。 

**主要过程：**

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgWxFyfLffOFf63WJUPsibwT4rva8At5qGzIU2FxQ9pOibJoSYpQ0Fz0ng/640?wx_fmt=other&from=appmsg)

**解释：**

+ 生成器语言模型 M
+ 检索器 R
+ 段落集合 {d1,d2,...,dN}

**输入：**接收输入提示 ( x ) 和之前生成的文本 （y<t） ，其中(y-t)是模型基于本次问题生成的文本？

**检索预测：**模型 ( M ) 预测是否需要检索（Retrieve），基于 ( (x, y<t) )。

**检索判断：**

如果 ( Retrieve ) == 是：

1.**检索相关文本段落：**使用 ( R ) 基于 ( (x, y<t) ) 检索相关文本段落 ( D )。

2.**相关性预测：**模型 ( M ) 预测相关性 ( ISREL )，基于 ( x )，段落 ( d ) 和 ( y<t ) 对每个 ( d ) 进行评估。

3.**支持性和有用性预测：**模型 ( M ) 预测支持性 ( ISSUP ) 和有用性 ( ISUSE )，基于 ( x, y<t, d ) 对每个 ( d ) 进行评估。

4.**排序：**基于 ( ISREL ), ( ISSUP ), 和 ( ISUSE ) 对 ( y<t ) 进行排序。

如果 ( Retrieve ) == 否：

1.**生成下一个段落**：模型 ( M ) 基于 ( x ) 生成 ( y_t )。

2.**有用性预测：**模型 ( M ) 预测有用性 ( ISUSE )，基于 ( x, y_t ) 进行评估。

 ![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIg1BG6GgGE5T6KjrdP6DrdRWS3kbILu7RF5MUjUkMpibATD7JwKgDuXWQ/640?wx_fmt=other&from=appmsg) 

+ **传统RAG：**先检索固定数量的文档，然后将这些文档结合生成回答，容易引入无关信息或虚假信息。
+ **SELF-RAG：**通过自反思机制，按需检索相关文档，并评估每个生成段落的质量，选择最佳段落生成最终回答，提高生成的准确性和可靠性。

### CRAG
论文地址：[https://arxiv.org/pdf/2401.15884](https://arxiv.org/pdf/2401.15884) 

**纠错检索增强生成（Corrective RetrievalAugmented Generation, CRAG）**，旨在解决现有大语言模型（LLMs）在生成过程中可能出现的虚假信息（hallucinations）问题。 

核心思想：

+ **问题背景：**大语言模型（LLMs）在生成过程中难免会出现虚假信息，因为单靠模型内部的参数知识无法保证生成文本的准确性。尽管检索增强生成（RAG）是LLMs的有效补充，但它严重依赖于检索到的文档的相关性和准确性。
+ **解决方案：**为此，提出了纠错检索增强生成（CRAG）框架，旨在提高生成过程的鲁棒性。具体来说，设计了一个轻量级的检索评估器来评估检索到的文档的整体质量，并基于评估结果触发不同的知识检索操作。

具体方法：

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgOvZjd3ByTrN4bdsCXcibSiaesBMHgstL5hyt5fVfichZNia1jNb2WsnibFw/640?wx_fmt=other&from=appmsg) 

解释：

1、 **输入和输出：**

+ x 是输入的问题。
+ D 是检索到的文档集合。
+ y 是生成的响应。

2、 **评估步骤：**

+ scorei 是每个问题-文档对的相关性得分，由检索评估器 E 计算。
+ Confidence 是基于所有相关性得分计算出的最终置信度判断。

3、 **动作触发：**

+ 如果置信度为 CORRECT，则提取内部知识并进行细化。
+ 如果置信度为 INCORRECT，则进行网络搜索获取外部知识。
+ 如果置信度为 AMBIGUOUS，则结合内部和外部知识。

4、  **生成步骤：**

+ 使用生成器 G 基于输入问题 x 和处理后的知识 k 生成响应 y。

关键的组件：

**1、检索评估器：**

+ 基于置信度分数，触发不同的知识检索操作：Correct（正确）、Incorrect（错误）和Ambiguous（模糊）。

**2、知识重组算法：**

+ 对于相关性高的检索结果，设计了一个先分解再重组的知识重组方法。首先，通过启发式规则将每个检索到的文档分解成细粒度的知识片段。
+ 然后，使用检索评估器计算每个知识片段的相关性得分。基于这些得分，过滤掉不相关的知识片段，仅保留相关的知识片段，并将其按顺序重组。

**3、网络搜索：**

+ 由于从静态和有限的语料库中检索只能返回次优的文档，因此引入了大规模网络搜索作为补充，以扩展和增强检索结果。
+ 当所有检索到的文档都被认为是错误的时，引入网络搜索作为补充知识源。通过网络搜索来扩展和丰富最初获取的文档。
+ 输入查询被重写为由关键词组成的查询，然后使用搜索API（如Google Search API）生成一系列URL链接。通过URL链接导航网页，转录其内容，并采用与内部知识相同的知识重组方法，提取相关的网络知识。

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIg05yLqVS52Zcf9egJrvwPza1RM39LEDmbOUBNLdms9p1VU8OvmwBHoQ/640?wx_fmt=other&from=appmsg) 

**实验结果**

在四个数据集（PopQA、Biography、PubHealth和Arc-Challenge）上的实验表明，CRAG可以显著提高RAG方法的性能，验证了其在短文本生成和长文本生成任务中的通用性和适应性。 

### Dense X Retrivel
论文地址：[https://arxiv.org/pdf/2312.06648](https://arxiv.org/pdf/2312.06648) 

探讨在密集检索任务中，选择适当的检索单元粒度（如文档、段落或句子）的重要性，作者提出了一种新的检索单元“命题”（proposition），并通过实验证明，在密集检索和下游任务（如开放域问答）中，基于命题的检索显著优于传统的段落或句子检索方法。 

**命题定义：**命题被定义为文本中的原子表达，每个命题包含一个独立的事实，并以简洁、自包含的自然语言形式呈现。命题代表文本中的原子意义表达，定义如下：

a.**独立意义：**每个命题应对应文本中的一个独立意义，所有命题的组合应能表示整个文本的语义。

b.**最小化：**命题应是最小的，即不能进一步分割成更小的命题。

c.**上下文自包含：**命题应是上下文化且自包含的。即命题应包含所有必要的上下文信息，以便独立理解其意义。 

具体的方法：

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgASDxicK8mP6aoC0230ibicsjYuWiavA1iaE2szxrOs2GlKRR1Yjw38rLcng/640?wx_fmt=other&from=appmsg)

具体的步骤：

+ **分割段落为命题**（A部分）
+ **创建FactoidWiki**（B部分）(命题化处理器将段落分割成独立的命题，从而创建了一个新的检索语料库FactoidWiki。)
+ **比较不同粒度的检索单元**（C部分）
+ **在开放域问答任务中使用**（D部分）

| <font style="color:rgb(62, 62, 62);">不同粒度的检索单元（段落、句子、命题）进行了实证比较</font> |
| --- |
| ![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgM8MBrwFtwOzUzhDuTzl4mLvZo6VCsb1m4E9QpyjkGPLOLdMTKU04GQ/640?wx_fmt=other&from=appmsg)![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgM8MBrwFtwOzUzhDuTzl4mLvZo6VCsb1m4E9QpyjkGPLOLdMTKU04GQ/640?wx_fmt=other&from=appmsg) |
| + <font style="color:rgb(62, 62, 62);">（Recall@5），即在前5个检索结果中找到正确答案的比例</font><br/>+ <font style="color:rgb(62, 62, 62);">（EM@100），即在前100个字中找到正确答案的比例</font><br/>+ <font style="color:rgb(62, 62, 62);">图表明了命题检索在提高密集检索的精确性和问答任务的准确性方面的优势。</font><font style="color:rgb(62, 62, 62);">通过直接提取相关信息，命题检索能更有效地回答问题并提供更准确的检索结果。</font><br/>+ <font style="color:rgb(62, 62, 62);">将段落和句子分解为更小的、上下文自包含的命题会产生更多的独立单元，每个单元都需要存储在索引中，会导致存储增加。</font> |


从段落到命题的提示词：(英文版)

```plain
Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.

1. Split compound sentences into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifiers to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
4. Present the results as a list of strings, formatted in JSON.

**Input:**

Title: Ìostre. Section: Theories and interpretations, Connection to Easter Hares. 

Content: 
The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in 1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were frequently seen in gardens in spring, and thus may have served as a convenient explanation for the origin of the colored eggs hidden there for children. Alternatively, there is a European tradition that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and both occur on grassland and are first seen in the spring. In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe. German immigrants then exported the custom to Britain and America where it evolved into the Easter Bunny."

**Output:**

[
  "The earliest evidence for the Easter Hare was recorded in south-west Germany in 1678 by Georg Franck von Franckenau.",
  "Georg Franck von Franckenau was a professor of medicine.",
  "The evidence for the Easter Hare remained unknown in other parts of Germany until the 18th century.",
  "Richard Sermon was a scholar.",
  "Richard Sermon writes a hypothesis about the possible explanation for the connection between hares and the tradition during Easter.",
  "Hares were frequently seen in gardens in spring.",
  "Hares may have served as a convenient explanation for the origin of the colored eggs hidden in gardens for children.",
  "There is a European tradition that hares laid eggs.",
  "A hare’s scratch or form and a lapwing’s nest look very similar.",
  "Both hares and lapwing’s nests occur on grassland and are first seen in the spring.",
  "In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.",
  "German immigrants exported the custom of the Easter Hare/Rabbit to Britain and America.",
  "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in Britain and America."
]

---

**Input:**
<an new passage>

**Output:**﻿
```

从段落到命题的提示词：(中文版)

```plain
将“内容”分解成清晰简洁的命题，确保它们在上下文之外也能理解。

1. 将复合句拆分为简单句。尽可能保持输入的原始措辞。
2. 对于任何伴随有额外描述信息的命名实体，将这些信息分离成独立的命题。
3. 通过为名词或整句添加必要的修饰词，并用完整的实体名称替换代词（例如“它”、“他”、“她”、“他们”、“这个”、“那个”）来使命题脱离上下文。
4. 以JSON格式将结果呈现为字符串列表。

**输入:**

标题: Ìostre. 部分: 理论和解释，与复活节兔子的联系。

内容: 
关于复活节兔子（Osterhase）的最早证据是在1678年由医学教授Georg Franck von Franckenau在德国西南部记录的，但在18世纪之前在德国其他地区仍然不为人知。学者Richard Sermon写道：“兔子经常在春天的花园中出现，因此可能是对为孩子们藏在那里的彩蛋起源的一个方便解释。或者，有一种欧洲传统认为兔子下蛋，因为兔子的抓痕或形态和大鸻的巢非常相似，而且两者都出现在草地上，并在春天首次出现。19世纪，复活节卡片、玩具和书籍的影响使复活节兔子在整个欧洲流行起来。然后，德国移民将这种习俗传播到英国和美国，并在英国和美国演变成复活节兔子。”

**输出:**

[
  "关于复活节兔子的最早证据是在1678年由Georg Franck von Franckenau在德国西南部记录的。",
  "Georg Franck von Franckenau是一位医学教授。",
  "在18世纪之前，复活节兔子的证据在德国其他地区仍然不为人知。",
  "Richard Sermon是一位学者。",
  "Richard Sermon写了一个关于兔子与复活节传统之间可能联系的假设。",
  "兔子经常在春天的花园中出现。",
  "兔子可能是对为孩子们藏在花园里的彩蛋起源的一个方便解释。",
  "有一种欧洲传统认为兔子下蛋。",
  "兔子的抓痕或形态和大鸻的巢非常相似。",
  "兔子和大鸻的巢都出现在草地上，并在春天首次出现。",
  "19世纪，复活节卡片、玩具和书籍的影响使复活节兔子在整个欧洲流行起来。",
  "德国移民将复活节兔子的习俗传播到英国和美国。",
  "复活节兔子的习俗在英国和美国演变成复活节兔子。"
]

---

**输入:**
\<新段落\>

**输出:**
```

## 其它优化方法
### Chunking 优化
视频地址：[https://www.youtube.com/watch?v=8OJC21T2SL4&t=1933s](https://www.youtube.com/watch?v=8OJC21T2SL4&t=1933s) 

主要观点：

+ 文本切割是优化语言模型应用性能的关键步骤。
+ 切割策略应根据数据类型和语言模型任务的性质来定制。
+ 传统的基于物理位置的切割方法（如字符级切割和递归字符切割）虽简单，但可能无法有效地组织语义相关的信息。
+ 语义切割和基因性切割是更高级的方法，它们通过分析文本内容的语义来提高切割的精确度。
+ 使用多向量索引可以提供更丰富的文本表示，从而在检索过程中提供更相关的信息。
+ 工具和技术的选择应基于对数据的深入理解以及最终任务的具体需求。

五种文本切割的层次：

1.**字符级切割：** 简单粗暴地按字符数量固定切割文本。

2.**递归字符切割：** 考虑文本的物理结构，如换行符、段落等，逐步递归切割。

3.**文档特定切割：** 针对不同类型的文档（如Markdown、Python代码、JavaScript代码和PDF），使用特定的分隔符进行切割。

4.**语义切割：** 利用语言模型的嵌入向量来分析文本的意义和上下文，以确定切割点。

5.**基因性切割：** 创建一个Agent，使用Agent来决定如何切割文本，以更智能地组织信息。 

### Query重写
参考文章：[https://raghunaathan.medium.com/query-translation-for-rag-retrieval-augmented-generation-applications-46d74bff8f07](https://raghunaathan.medium.com/query-translation-for-rag-retrieval-augmented-generation-applications-46d74bff8f07)  

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIg8YswrnBmQd9vPD7azu90XAuwRIr1LlRUkzqTWyfxeCWFpliaISyHsKg/640?wx_fmt=other&from=appmsg) 

**1、输入问题：**

**2、生成子查询：**输入问题被分解成多个子查询（Query 1, Query 2, Query n）。

**3、文档检索：**每个子查询分别检索相关的文档，得到对应的检索结果（Query 1 documents, Query 2 documents, Query n documents）。

**4、文档合并和重新排序：**将所有子查询的检索结果文档合并并重新排序（Combined Reranked Documents）。

**5、输出答案：**

提示词：

```plain
你是一名乐于助人的助手，负责生成与输入问题相关的多个子问题。
目标是将输入问题分解为一组可以单独回答的子问题。
生成与以下问题相关的多个搜索查询：{question} 
输出（3个查询）：
```

英文版：

```plain
You are a helpful assistant that generates multiple sub-questions related to an input question. 
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. 
Generate multiple search queries related to: {question} 
Output (3 queries):
```

上面是分解为多个子查询之后，把多个子查询召回的文档重排序后交给大模型来回答，接下来的策略对子查询生成的问题进行迭代得到新的答案。 

其它Query分解的策略：

1、递归回答方法：我们将问题以及先前的问答响应和为当前问题获取的上下文继续传递。保留了旧的视角，并将解决方案与新的视角同步，这种方法对于非常复杂的查询是有效的。

 ![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgYOtUjyR5qHp3Fm8hsIr1EmBTKfeKia6hr1IicqZn7Y5IUI1iaBm5zxGaw/640?wx_fmt=other&from=appmsg)  

2、并行回答方法：将用户提示分解为细致入微的部分，不同之处在于我们试图并行解决它们。单独回答每个问题，然后将它们组合在一起以获得更细致的上下文，然后用于回答用户查询。对大部分场景来说是有效的方案

 ![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgLvboMOVxkHcibMHvG6htB4EIQWJ2fZXlq4ct9JreGajawFvoBBsWFmQ/640?wx_fmt=other&from=appmsg)  

其它：

 ![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgzENiaicjxlelamQq5xG5nCcpeibevDysDFqJeCbkWjhsyNThShvOHDvcg/640?wx_fmt=other&from=appmsg) 

**1.分析Query结构（分词、NER）**

+ 分词：将查询分解成单独的词语或词组，便于进一步处理。
+ NER（命名实体识别）：识别查询中的命名实体，如人名、地名、组织等，有助于理解查询的具体内容。

**2.纠正Query错误（纠错）**

+ 对查询中的拼写错误或语法错误进行自动纠正，以提高检索的准确性。

**3.联想Query语义（改写）**

+ 对查询进行语义改写，使其更具表达力和检索效果。
+ HyDE：一种基于上下文的联想查询改写方法，利用现有信息生成更有效的查询。
+ RAG-Fusion：结合检索和生成的技术，进一步增强查询的表达能力。

**4.扩充Query上下文（省略补全、指代消解）**

+ 对查询进行上下文扩展，补全省略的内容或解析指代关系，使查询更完整和明确。

 ![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgUhOy1KSHoDHibURvJnBTrZ5Belp636o6mYLSGHsyrRCVKy6XIpB5Gaw/640?wx_fmt=other&from=appmsg)

### Hybrid Retrieval（混合检索）
混合检索(Hybrid Retrieval)是一种结合了稀疏检索(Sparse Retrieval)和稠密检索(Dense Retrieval)的策略，旨在兼顾两种检索方式的优势，提高检索的效果和效率。

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgBE0S2QyYTwVia8HicpgS2Wrt2oB7RiaPiaH8yoQRNiaBHradkK92hT8LDuA/640?wx_fmt=other&from=appmsg)

1.稀疏检索(Sparse Retrieval)：这种方法通常基于倒排索引(Inverted Index)，对文本进行词袋(Bag-of-Words)或TF-IDF表示，然后按照关键词的重要性对文档进行排序。稀疏检索的优点是速度快，可解释性强，但在处理同义词、词语歧义等语义问题时效果有限。

2.稠密检索(Dense Retrieval)：这种方法利用深度神经网络，将查询和文档映射到一个低维的稠密向量空间，然后通过向量相似度(如点积、余弦相似度)来度量查询与文档的相关性。稠密检索能更好地捕捉语义信息，但构建向量索引的成本较高，检索速度也相对较慢。 

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgNqo3V7L2RDCD4XhYD5YroRP8ibEnQrKba9S9UGF5EjI2x3htx2ib6VGA/640?wx_fmt=other&from=appmsg)

### Small to Big
Small to Big检索策略是一种渐进式的多粒度检索方法，基本思想是，先从小粒度的文本单元(如句子、段落)开始检索，然后逐步扩大检索范围到更大的文本单元(如文档、文档集合)，直到获得足够的相关信息为止。 

感觉可以和上面的Dense X Retrivel结合；

"Small to Big"检索策略的优点在于：

1.提高检索效率：通过先检索小粒度单元，可以快速锁定相关信息的大致位置，避免了在大量无关文本中进行耗时的全文搜索。

2.提高检索准确性：通过逐步扩大检索范围，可以在保证相关性的同时，获取更完整、更全面的上下文信息，减少由于语义片段化而导致的检索不准问题。

3.减少无关信息：通过从小到大的渐进式检索，可以有效过滤掉大部分不相关的文本，减少无关信息对后续生成任务的干扰。 

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIg3Lk6tDhic6mxdRKNKKk2MZCsJB9uGP4mCdG4bFeMIXvoNKrXPzItxrA/640?wx_fmt=other&from=appmsg)

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIglAXH35tJCdDEyZibT3icp3WnjjXEnn6KTlQNSzKgvKatO1b7RRr4ONrA/640?wx_fmt=other&from=appmsg) 

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgCK8OQ0Micd8pn6icxCLCW4Gic3oqZKov4ZcCFzqQUZtCialH2ianTeeq1Tg/640?wx_fmt=other&from=appmsg)

### Embeding&Rerank模型优化
![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgicJlBbRbPZdg4AbCQDLzfhOaaXj5RGDUYp1ia6Ib5TRC3SYjRItemJXA/640?wx_fmt=other&from=appmsg) 

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgawvJOq2uN9WC39r2crSZTicYvz50JBqjNWVmJccb0lXFwGnpmhZ5Qew/640?wx_fmt=other&from=appmsg)

 ![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgMUHDTT3jxgJ5uc78FvNOtTQSJ93jDXedbIiagvZbxjauiaYNxdHlEROQ/640?wx_fmt=other&from=appmsg)

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgTY621kHf2pV2mib2icorUJicGvqZicaRicwIS04gUUbgoJHRC6N40qEIbBg/640?wx_fmt=other&from=appmsg)

## 效果评估 
RAG评估系统是模型开发、优化和应用过程中不可或缺的一部分，通过统一的评估标准，可以公平地比较不同RAG模型或优化方法之间的差异，识别出最佳实践。

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgevpibs4fvRozibQFeBsY9ULGQkwHhu3mq4K0JKA7fT7F3CxpkWJULMRQ/640?wx_fmt=other&from=appmsg) 

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIggVSMkia15VyMHr1xtRsj6t2R8FoooeKxZRzRicLTkrqibMHaS6WpgYWNw/640?wx_fmt=other&from=appmsg) 

![](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIoeTG8MyTaZZ02jVe4aFIgWWMaB1BRDVULV4S0Hgn9FyZLcmOibiaFWXN0ibvPKMdEXwZDgpjITPttQ/640?wx_fmt=other&from=appmsg)  

## 最后
在RAG系统的实际应用中，需要工程和算法等的多方参与和努力，理论上有很多方法，在实践的过程中我觉得还需要大量的实验对比，不断验证和优化，也可能会遇到许多细节问题，比如可想到的异构数据源的加载和处理啊，知识的展示形态(文本、图片、表格)等是否能一起回答，提升下用户体验，以及建立一套自动化的评估机制，当然还有模型的持续迭代和大小模型的训练支持。

**参考链接：**

+ RAPTOR：[https://arxiv.org/pdf/2401.18059](https://arxiv.org/pdf/2401.18059)
+ Self-RAG：[https://arxiv.org/pdf/2310.11511](https://arxiv.org/pdf/2310.11511)
+ CRAG：[https://arxiv.org/pdf/2401.15884](https://arxiv.org/pdf/2401.15884)
+ Dense X Retrivel：[https://arxiv.org/pdf/2312.06648](https://arxiv.org/pdf/2312.06648)
+ The 5 Levels Of Text Splitting For Retrieval：[https://www.youtube.com/watch?v=8OJC21T2SL4&t=1933s](https://www.youtube.com/watch?v=8OJC21T2SL4&t=1933s)
+ RAG for long context LLMs：[https://www.youtube.com/watch?v=SsHUNfhF32s](https://www.youtube.com/watch?v=SsHUNfhF32s)
+ LLamaIndex构建生产级别RAG：[https://docs.llamaindex.ai/en/stable/optimizing/production\_rag/](https://docs.llamaindex.ai/en/stable/optimizing/production\_rag/)

