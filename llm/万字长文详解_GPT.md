2023 上半年科技与创投圈的最大热点无疑是**大模型及其相关技术**。自从 OpenAI 在去年 11 月底发布 ChatGPT，其表现出来的强大能力迅速震撼了科技从业者、创作者，并以历史最快的速度获取了超过 1 亿用户。

在接下来的 2023 年上半年，各大科技公司与科研机构关于大模型的发布令人眼花缭乱，大多数人才开始意识到原来以 GPT 为代表的一系列深度学习的技术已经走得如此之远。

如果我们点开 「State of AI」从 2018 年开始做的每年 AI 年度回顾报告的话，我们也许会进一步深刻意识到世界这一螺旋式发展的规律，在经历了上一波由 AlphaGo 引发的创投 AI 热潮与沉寂之后，以 Transformer 和 Diffusion Model 为代表的相关技术仍然在草蛇灰线地向前发展着，直到这一次才再度点燃世界。

## 01 概述
本文所有的资料来源于互联网的公开信息，更多是从技术视角去理解和梳理相关技术。

本文结构主要参考自《State of AI[1]》和《A Survey of Large Language Models[2]》，强烈推荐和鼓励大家去阅读本文附录的原始资料，希望本文可以作为众多程序员们学习 GPT 相关技术的一个资料索引。

理解 Transformer 等模型要求人们了解从 RNN/LSTM 到 Transformer 这一发展的历史源流，并且具备基本的数学功底和耐心去学习相关知识。本文尽可能从程序员的视角去解释其基本原理，期望读者已经体验过类似于 ChatGPT 或者 Stable Diffusion 等相关产品，也期望读者对于神经网络和反向传播有着基本的认识 （推荐吴恩达的深度学习课程作为入门[3]）。

本文的重点不是模型和算法原理，不会去解释从 Transformer 到 BERT 和 GPT 之间的路线变化，而是尝试从程序员的角度理解 GPT 这一技术的基本组成，尝试去了解那些大家都在谈论的名词背后的基本含义，乃至当前生机勃勃的 AI 应用的发展。

本文主要关注类似于 ChatGPT 和 LLaMA 这种大语言模型，暂时不会涉及类似 Stable Diffusion 等语音、图片和视频领域，乃至其他的多模态场景。

作为一个软件工程师，本文作者对于很多算法的理解也并不算深刻与全面，甚至可能会存在理解偏差和错误，在介绍相关方向的时候也肯定会有遗漏，欢迎大家交流与指正。

## 02 技术术语
在真正开始之前，这里先简单介绍下本文可能会碰到的技术名词，现在不需要深刻理解其含义，只需要有初步印象即可。

| 英文 | 中文 | 解释 |
| :---: | :---: | :---: |
| Fine Tuning | 微调 | 将已经训练的模型的参数作为新模型的初始化参数重新训练 |
| RLHF | 基于人类反馈的强化学习 | 让人给模型生成的结果打分，用人打的分来调整模型 |
| Alignment | 对齐 | 让机器生成复合人类期望的，复合人类价值观的语句 |
| Scaling Laws | 扩展定律 | 模型效果的线性增长要求模型的大小指数增长 |
| Emergent Ability | 涌现能力 | 小模型没有，只有模型大到一定程度才会出现的能力 |
| In-Context Learning | 上下文学习 | 在 Prompt 里面写几个例子，模型就可以照着这些例子做生成 |
| Chain-of-Thought | 思维链 | 在写 Prompt 的时候，不仅给出结果，还要一步一步地写结果是怎么出来的 |
| Prompt Engineering | Prompt 工程 | 关注提示词开发和优化，帮助用户将大语言模型用于各场景和研究领域 |
| LLM | 大语言模型 | 模型规模和训练的数据规模都很大的语言模型 |
| Agent | 智能体或者智子 | 基于 LLM 能够自主行为的智能体 |
| LoRA | 低秩自适应 | 一类旨在通过低维结构近似大模型的高维结构来降低其复杂性的技术 |
| Vector Database | 向量数据库 | 一种专门用于存储和查询向量数据的数据库系统 |
| ZeRO | 零冗余优化器 | 一种针对大规模分布式深度学习的新型内存优化技术 |


## 03 Hello World
基本上所有程序员学习编程语言的第一课是 Hello World。体验过 ChatGPT 的强大能力后，作为程序员的你或许会好奇这背后到底是如何构建出来的。幸运的是，除了 OpenAI 的 GPT 模型，其他的很多公司也发布很多的开源或闭源大语言模型 LLM。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92wZFicnM8FnTLtrvwKjqRFBJGvy47G3sGtGKfMCbYZzJJuuSHnmdICyA/640?wx_fmt=png)

LLM Timeline， [https://github.com/RUCAIBox/LLMSurvey](https://github.com/RUCAIBox/LLMSurvey)

其中的优秀代表就是 Meta 在 2023 年发布的 LLaMA 模型，开源社区围绕着 LLaMA 这一开源模型（只开源了模型，权重被「泄漏」）构建了丰富的生态。在 GitHub 上有很多基于 LLaMA 的开源项目，其中 llama.cpp 和 Chinese-LLaMA-Alpaca 就可以作为我们学习和了解 LLM 的 Hello World。

本文并不打算详细介绍如何基于 llama.cpp 和 Chinese-LLaMA-Alpaca 构建自己的 ChatGPT 的具体步骤。程序员最大的优势即在于他们快速的学习能力和强大的动手能力，参考 Chinese-LLaMA-Alpaca 这一项目的 Wiki 和网上的公开资料，不需要昂贵的 GPU，你就可以快速在自己的笔记本上构建自己的 ChatGPT。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92Dxs9mIxPqW4GNsXicygQ22KhknIpTdKHx1KicNClOJaVrE1zhDSqHPCQ/640?wx_fmt=png)

在这个 Hello World 中，你可以体验到：

+  基于已有的 LLaMA 模型和计算设备，便可以开始最简单的 NLP 推理任务 
+  LLaMA 作为预训练模型，可能并不太适合中文场景的相关任务，需要通过扩充中文词表和二次预训练来进一步提升中文基础语义理解能力 
+  预训练之后，还需要进一步通过中国指令数据进行 Fine-Tuning，从而进一步提升指令理解（问答、写作、建议等）和多轮上下文理解的能力（聊天）等 
+  可以通过 LoRA 等技术合并预训练模型，减少 Fine-Tuning 的成本 
+  即使是 LLaMA-7B 也需要强大的算力，通过 llama。cpp 可以提升推理的效率，如果有 GPU 则可以进一步加速推理 
+  本项目还可以进一步结合 LangChain 和 Transformers 等开源组件，基于 Vector Database 开发类似于 privateGPT 这种本地知识库 QA 系统 

除了这些之外，你可能还会好奇：

原始的 LLaMA 模型结构如何，是如何在原始的 Transformer 结构上长出这些大模型的？

类似于 LLaMA/GPT4 这些大语言模型，从原始的数据和模型结构，到最终类似于 ChatGPT 这种可用的 Assistant 模型，需要经历哪些阶段，会应用到哪些技术？

当我们已经看到大语言模型的巨大潜力之后，我们可以在各种垂类场景或者是应用层做哪些创新呢？

类似于 AutoGPT8 这种 Intelligent Agent 生态正在蓬勃发展，这对于以后的软件开发又意味着什么呢？

## 04 Baking The Model
在实际上手实践过 Hello World 之后，你开始会对大语言模型有了一些基础的认知，接下来我们会深入到其内部，了解 GPT 这种大语言模型的菜到底是怎么做出来的。

OpenAI 的 Andrej Karpathy 在最近 Microsoft Build 2023 对 GPT 的菜谱做了非常详细的介绍，强烈推荐阅读 State of GPT[4]。

下图展示了当前 （2023） GPT Assistant Model 的训练流水线，这个 Pipeline 正在快速发展中，但也可以从这张图一窥全豹：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92EGYibCkcCcibmYwab6gSIyoRRwJYDYpdglyONV3NnIA8kRTEOBgPehAg/640?wx_fmt=png)

GPT Assistant Training Pipeline，[https://karpathy.ai/stateofgpt.pdf](https://karpathy.ai/stateofgpt.pdf)

### 预训练
自从 2017 年 Transformer 的论文 Attention is All You Need 发布之后，根据 Encoder/Decoder 的路线不同，各种不同的 Base Model 已经发布。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92GO6JKicx5jRFQZW8Sb2NUsNbmRX137Q6UtMXQ00uJOuGU7bQRCK7ASg/640?wx_fmt=png)

LLM Evolutionary Tree，[https://github.com/Mooler0410/LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide)

**预训练是大语言模型获取能力的基础。**在预训练过程中，大规模高质量预训练语料的获取、优秀的模型结构设计和高效稳定的并行加速训练优化技术这三点都非常重要。

#### 1、数据采集
预训练语料的规模和质量对于 LLM 模型容量和能力至关重要。当前 LLM 主要混合各种公共文本数据集作为预训练语料，可以广义的分为通用文本数据和专用文本数据：

+  通用文本：包含网页、书籍和对话文本等，规模大、多样性强且易于获取，为大多数 LLM 所使用 
+  网页：包括 WikiPedia 这些高质量的文本和 CommonCrawl 这种相对低质量的文本，因此过滤和处理网页以提供数据质量十分重要 
+  对话文本：包括 Reddit Links 或者从其他社交媒体收集的对话数据 
+  书籍：包括像 Gutenberg 甚至 Z-Library 中获取的书籍数据 
+  专用文本：包含多语言文本、科学文本和代码等，对于提高 LLM 在翻译、多语言问答和代码生成等特定下游任务重的能力非常有用 
+  多语言文本：包含多语言语料库以增强多语言的理解与生成能力 
+  科学文本：一般来自 arXiv 论文、科学教材、数学网页和其他的科学资源 
+  代码：一般来自类似于 StackExchange 这种编程问答社区和 GitHub 这种公共代码仓库 

下图展示了一些代表性 LLM 的预训练预料数据来源的分布情况：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92fXnGj5ic3kXbn4YG16GUJ0rMlwWpcTV3XpszoJ3p1zoNrF5pTeYXlYA/640?wx_fmt=png)

Data mixture for pre-training LLMs，[https://github.com/Mooler0410/LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide)

在收集大量文本数据之后，需要对数据进行预处理，特别是消除噪声、冗余、无关和潜在有害的数据。如下图所示，预处理 LLM 的预训练数据的典型流程大致包括：

+  Quality Filtering：基于语言、Metric、统计和关键词进行过滤 
+  De-duplication：重复数据可能会降低 LLM 的多样性，因此需要在句子级、文档级和数据集级等不同粒度进行去重 
+  Privacy Reduction：来自网络的预训练文本数据可能会涉及敏感或者个人信息的用户生成内容，可能会增加隐私泄漏的风险，因此需要从预训练语料中删除可识别个人信息 
+  Tokenization：分词将原始文本分割成词序列，通常使用 SentencePiece 作为预训练语料库训练定制化的分词器，同时利用字节级 Byte Pair Encoding （BPE）确保分词后信息不丢失 

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92HDy33O4oVyqwawfW78fYibEZwDe3N40pEib95BxJ7ncgZNicuoERI5y1A/640?wx_fmt=png)

Typical data preprocessing pipeline for pre-training LLMs，[https://github.com/Mooler0410/LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide)

下图直观展示了 Tokenization 的过程，分词会将文本分割成词序列，最终以整数组成的词向量输入给模型。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92hC5IkfXepZTDu8dMovREC4pNYTqib5CdzsYxaZibRNLIiblZ5voibsuy5w/640?wx_fmt=png)

Tokenization， [https://karpathy.ai/stateofgpt.pdf](https://karpathy.ai/stateofgpt.pdf)

以 LLaMA-65B 为例，使用 2048 张 A100 的 GPU 卡，基于 32k 大小的词表，在 1.4T 的 Tokens 上训练了 21 天。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92wpjyeoicoq3O8EibzqqM3ZyIsoeWibI5UJhnuPoHicrpMt8icTv1xiaYO7Lg/640?wx_fmt=png)

Training Token Size，LLaMA：Open and Efficient Foundation Language Models

#### 2、模型架构
由于 Transformer 架构的出色并行性和容量，Transformer 架构已经成为开发各种 LLM 的事实标准，使得将语言模型扩展到数百亿或者数千亿参数成为可能。

下图是 Transformer 的经典架构图，作为之前没怎么接触过模型结构的程序员看到这张图可能会感到害怕，我们不需要非常详细的理解这个模型的原理，只需要知道 ——

标准的 Transformer 模型主要由两个模块构成：

+  Encoders（左边）：负责理解输入文本，为每个输入构造对应的语义特征 
+  Decoders（右边）：负责生成输出，使用 Encoder 输出的语义表示结合其他输入来生成目标序列 

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92mgwLfvMaGEjF2WCV6whxKreWUfYzrFeks4HhFd6rtLgaDLMJc2fVcg/640?wx_fmt=png)

The Transformer：Model Architecture

如果想要进一步了解 Transformer 这一架构的原理，强烈推荐阅读 Jay Alammar 的博客。从为了解决翻译问题时 Seq2Seq 模型的提出的 Attention 的基本概念，到 Transformer 架构中完全抛弃 RNN 提出的 Attention is All You Need，到 GPT-2 和 GPT-3 的架构解读，Jay Alammar 的博客都提供了精彩的可视化配图便于理解模型结构。

作为引用，我们再通过这些配图简单理解下 Transformer 的结构，以下的配图都来自 Jay Alammar 的博客：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92b6rlZDB5qXicdUyCx790b40GdryqLMHxGez6JlvmHAzvcJVHs97vOjA/640?wx_fmt=png)

Transformer 架构主要由 Encoders 和 Decoders 组成，一般会经历从输入到 Encoders 再到 Decoders，最终到输出

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92GsyMLoRpfbmGyerxNrqcQnvdXBNAYRuTzwjSJmFpgKTfnfHKgK2MbQ/640?wx_fmt=png)

进一步拆解 Encoders 和 Decoders，可以看到 Encoders 由一系列 Encoder 堆叠而成，Decoders 也是类似

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92URcuXL7GjoOHRK5vEhTb6XTDQgWnnKjgsagyDEN7NYu4jJW7ZLoOSw/640?wx_fmt=png)

所有的 Encoder 都具备相同的结构（尽管他们不分享权重），每一个 Encoder 都可以分成两层

+  Self-Attention：这即是 Transformer 中 Attention 之所在，通过 Self-Attention 可以让模型在处理文本时，将注意力只放在某些词语上 
+  Feed-Forward Neural Network：Feed-Forward 是一个两层的全连接层 

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz9226lYiagBVuwWnhBqxu2mxKYicJk2GevIIbvotqlVLNJbCcurIlGbFgAA/640?wx_fmt=png)

Decoder 的结构与 Encoder 的结构非常相似，只是这里多了一层 Masked Self-Attention

+ 这里的 Masked Self-Attention 的作用是将未来的 Tokens 给 Mask 掉，保证 Decoder 生成当前 Token 的概率分布时，只看到过去的信息，不看到未来的信息

看到这里，我们可以在回顾下本节开始的那张图片，根据 Transformer 中 Encoder 和 Decoder 的不同组成，我们可以把当前主流 LLM 架构分成三类：

+  Decoder-Only：典型代表是 GPT 系列和 LLaMA 等模型 
+  Encoder-Only：典型代表是 BERT 和 ALBERT 等模型 
+  Encoder-Decoder：典型代表是 T5 和 BART 等模型 

针对 GPT 系列这种 Decoder-Only 的架构，我们可以看到其架构组成：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92IN6UsHHFGXmU3kraoauKnQC50UOrKaRWgf2AKJK8hE8ZuYkgbJibvuw/640?wx_fmt=png)

落脚到业界实际的模型，我们可以看到 GPT-1 是 12 层的 Decoder-Only 的 Transformer 架构，如下图所示：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92NQDIzsyYo1Rsou73Zz0KGTupTIcsVQtcaicflv4HYucl9hsc90JnzwQ/640?wx_fmt=png)

GPT-1 Architecture

简单来说，GPT 系列基本可以理解成类似的架构，GPT-3 相对于 GPT-1 只是 Decoder 的层数更多，训练的预料数据更大。有意思的是，GPT-1 和 GPT-2 并没有展示出类似于 GPT-3 相同的强大能力，表明模型规模的增加在扩大模型架构的容量发挥了巨大的作用。Jason Wei 的这个演讲[5]对这种 Scaling Law 和 Emergent Ability 做了详细的讨论，你也可以参考符尧的这篇文章[6] 和张俊林的这篇文章[7]。

本文并不会尝试去解读 GPT 系列模型的具体结构，这个工作交给算法科学家会更加合适。李沐老师的「跟李沐学 AI」系列视频专栏[8]分析了整个 GPT 系列的相关论文，深入浅出，形象生动，强烈推荐！

总结：本小节简单阐述了从 Transformer 到 GPT 相关的模型结构，只是简要讲解了其组成，很多关于 Self-Attention 的具体计算机制、位置编码的原理与选择、激活函数的选择、Layer Norm 的相关原理都没有介绍，也推荐大家阅读前面附录的参考资料获得更加深刻的理解。

#### 3、模型训练
假设我们现在有了数据，也搞定了预算可以购买数千张当前 NVIDIA 最新款的 GPU，也有了设计好的模型结构，一切就绪，是不是就可以训练出一个大模型，一显身手了呢？

事情并没有那么简单。在当前如此巨大规模的模型和数据条件下，高效而稳定的训练 LLM 充满着挑战。HuggingFace 的 BigScience 团队为我们详细展示了这一训练过程中碰到的各种困难和训练所要求的软硬件工程和技术要点[9]。

本小节将简单阐述预训练过程中所依赖的一些技术，期望读者阅读后能够对这一领域有初步的了解。如果要详尽地解释其背后的原理与机制，则需要单独一篇文章来论述了。

3D 并行实际上是三种常用并行训练技术的组合，即数据并行、流水线并行和张量并行。

![3D Parallelism](https://mmbiz.qpic.cn/sz_mmbiz_jpg/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz9291pq3sNRyI07sAvdPKH2PcmmsiaJvDibDs5S76djEx9HiaXgibV3bvqbDA/640?wx_fmt=jpeg)

**数据并行**：将模型参数和优化器状态复制到多个 GPU 上，每个 Worker 并行处理数据的一部分，执行前向和反向传播以获取梯度。在不同 GPU 上计算的梯度将进一步聚合以获得整个批量的梯度，从而更新所有 GPU 上的模型。比如早期的 Parameters Server 和 Ring-AllReduce 都是这种原理。

然而这种数据并行的方法有一个限制，即是**要求当模型可以放进单个 GPU 时才有效**。当前 LLM 模型规模已经突破了这个限制，以 Bloom 模型为例，其参数量为 176B，如果是 float16 来表示参数，每个参数 2 个字节，也有 350GB 的大小了。而当前 NVIDIA 最新款 H100 的最高配置也只有 188 GB，想把模型放进单个 GPU 几乎已经不太可能。

为了解决这个问题，微软提出了 ZeRO，这篇文章[10]非常详尽解释了 ZeRO 3 个 Stage 的优化措施，并且有专门的视频讲解，最终可以用下面这张图来表述。对于这张图不必害怕，ZeRO 的思路很简单，就是普通的数据并行，只是每个 GPU 没有都复制完整的模型参数、梯度和优化器状态，而是只存储其中的一部分。在之后的训练过程，当需要给定层的完整层参数时，所有 GPU 通过通信同步以相互提供它们缺失的部分。

![ZeRO](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92fIkhAAoIF6LWjtwStOCYRfFmTiakWaevzVUKMia6XYb3rDouQMQIibianQ/640?wx_fmt=png)

**流水线并行**：将模型的不同 layer 分配到多个 GPU 中，数据在不同层（也即是不同 GPU ）之间移动。下图中 a 展示了流水线并行的计算方式，前向传播时，数据从 Device0 依次传递到 Device3；然后经历反向传播，计算梯度从 Device3 依次传递回 Device0。

b 展示了朴素流水线并行的方式，因为每个 GPU 必须等待前一个 GPU 计算完成，从而导致不必要的气泡开销。为了减少气泡开销，GPipe 和 PipeDream 提出如果同时进行多个迭代，每个节点在同一时刻负责不同迭代的计算，就可以避免数据依赖，不用在原地干等了。

![Naive Pipeline Parallelism vs Gpipe](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92mmZrdjtKE1upKhfLRPowaTEGTtl0SMTy3SkqJvaVofyc9ibIrRyLtAg/640?wx_fmt=png)

**张量并行**：张量并行把全连接层的参数和计算分割到多个 GPU 上，与流水线不同，张量并行专注于分解模型的张量（参数矩阵），Megatron-LM 论文[11]中关于这个问题有详细阐述。

![Tensor Parallelism](https://mmbiz.qpic.cn/sz_mmbiz_jpg/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92FhDnbTDABGGuRjLT1O6ib10dTxNlqwk39UZgL3B7XezVKvGJPFP75qg/640?wx_fmt=jpeg)

除了这里所介绍的一些并行训练的方法，还有类似于混合精度训练等技术，此处不在详细介绍。

千卡级别的模型训练充满着挑战，**经常会碰到训练不稳定导致模型崩溃的情况**。而这种情况下的故障很可能来自多个方面，可能是硬件的故障、PyTorch 的死锁、甚至是其他方面的各种问题。

由于训练是全天候 24/7 进行的，我们需要有人随叫随到 - 但由于我们在欧洲和加拿大西海岸都有人，因此不需要有人携带传呼机，我们能很好地互相备份。当然，周末的训练也得有人看着。我们自动化了大部分事情，包括自动从硬件崩溃中恢复，但有时仍需要人工干预。

除此之外，你还需要做好集群层面机器学习负载的灵活调度，在这方面，Anyscale 的 Ray 和微软的 Singularity 也做了很多的工作。

### 监督微调
经过预训练之后，我们可以获得类似于 LLaMA 这样的 Base Model，但这距离 ChatGPT 这种 Assistant Model 还有一段距离。Base Model 并不能像 Assistant Model 一样能够很好的适用于指令理解、多轮聊天和 QA 问答等场景，而只是总是倾向于去续写文本，如下图所示：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz927rcmvXwfZPBLOdB0UBs2N56w4K3rBnsib1m9aTwZEricvTzqw5ZuwkAQ/640?wx_fmt=png)

Base models are NOT Assistant，[https://karpathy.ai/stateofgpt.pdf](https://karpathy.ai/stateofgpt.pdf)

也即是说，Base Model 并不能很好的和用户的意图对齐 （Align），有可能会生成无用、甚至是有害的内容。为了解决这个问题，算法科学家们提出了很多的解决方案，典型代表就是以 InstructGPT 这种先经过 Supervised Finetuning，然后通过 Reward Modeling 和基于人类反馈的强化学习 RLHF。

![InstructGPT](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92aRmwkreKn4aAo5iaW3qyEtuLT7nUkftWU1TLl0o68LrurnlPY9tlt6w/640?wx_fmt=png)

DeepSpeed 团队也发布了 DeepSpeed-Chat，开源了一个支持端到端的 RLHF 的训练和推理系统，复刻了 InstructGPT 论文中的训练模式，并确保三个阶段一一对应：

+  监督微调，SFT 
+  奖励模型微调，RM 
+  基于人类反馈的强化学习，RLHF 

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92recPZm6F0HJnhicHtsw93g08Ztvcp5cvoSKG6ySLbDGcaGdVVN9xIoQ/640?wx_fmt=png)

如果想通过源码理解这个过程到底发生了什么，强烈推荐阅读 DeepSpeed-Chat 的相关代码。

SFT，或者也可以称作 Instruction Tuning，是一种有监督的任务训练，可以**使 LLM 展现出泛化到未见过任务的卓越能力**，即使在多语言场景下也能有不错表现。为了进行指令微调，我们需要一批高质量的 Instruction 数据集，比如 OASST1 。通常情况下，一个 Instruction 格式的实例包含一个任务描述、一对输入 - 输出以及少量示例（可选）。当前 Instruction 数据集可以来自以下几种方式：

+  格式化已有数据集：将已有的语料数据集转化成 Instruction 格式 
+  格式化真实人类需求：基于用户的真实需求而来，比如 OpenAI 基于其真实用户提交给 OpenAI API 的查询作为任务描述，这样一般效果更好 
+  格式化合成数据集：为了减轻人类在构建 Instruction 数据集中的负担，这种方式将现有实例输入到 LLM 生成多样的任务描述和实例来构建实例 

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92gCxZx6nPDyhIq8VsCSG4lrFo7N6AD3zJW8MBuvl1aANacUNoiamuYoA/640?wx_fmt=png)

构建 Instruction 数据集的不同方法，[https://github.com/RUCAIBox/LLMSurvey](https://github.com/RUCAIBox/LLMSurvey)

一般来说，增加指令的多样性、提供 CoT Prompting、增加多任务的任务数量，都可以增加 LLM 模型对于 Instruct 的理解能力。

### 奖励建模
RM 的训练是 RLHF 区别于旧范式的开端。这一模型接收一系列文本并返回一个标量奖励，数值上对应人的偏好。我们可以用端到端的方式用 LM 建模，或者用模块化的系统建模 （比如对输出进行排名，再将排名转换为奖励） 。这一奖励数值将对后续无缝接入现有的 RL 算法至关重要。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92SnRiaoo8bTXyJpqXzjM3w0C5gh7FJ8nOZKozChffOQS8rxb4Y2hPbSQ/640?wx_fmt=png)

Reward Model， [https://huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)

这里在训练奖励模型的时候，不是直接对文本标注分数来训练奖励模型，是**因为标注者的价值观不同导致这些分数未经过校准而充满着噪音**。对具体的排名方式，一种成功的方式是对不同 LM 在相同提示下的输出进行比较，然后使用  Elo 系统建立一个完整的排名。这些不同的排名结果将被归一化为用于训练的标量奖励值。

### 强化学习
简单总结下强化学习的基本概念：

+  强化学习的目标是训练一个智能体 Agent，使其在一个未知的环境 Enviroment 中完成任务。Agent 从环境中接收观测 Observations 和奖励 Reward，并向环境发送动作 Action。Agent 由策略 Policy 和学习算法两个部分组成。 
+  Policy 是一种映射，它基于环境的 Observations 选择 Action。通常 Policy 是一个带有可调参数的函数逼近器，比如深度神经网络。 
+  学习算法根据 Action、Observation 和 Reward 不断更新 Policy 的参数，它的目标是找到一种最优的 Policy，使得在任务期间累计获得的奖励最大化。 

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92dcHWcM6RorjuInqZzfegXRmPCQexiaS6iaVJVibbKFanxLa31CWlTRjdw/640?wx_fmt=png)

Reinforcement Learning， [https://es.mathworks.com/help/reinforcement-learning/ug/what-is-reinforcement-learning.html](https://es.mathworks.com/help/reinforcement-learning/ug/what-is-reinforcement-learning.html)

对于 LLM 强化学习来说：

+  策略 Policy 是一个接受提示并返回一系列文本的 LM，也就是我们微调的大模型 
+  动作空间 Action Space 就是 LM 的词表对应的所有词元，一般在 50k 数量级 
+  观察空间 Observation Space 就是可能的输入词元序列，也比较大（词汇量 ^ 输入标记的数量） 
+  奖励函数 Reward Function 由第二步的奖励模型和策略约束结合 
+  学习算法 RL Algorithms 目前用的比较多的是策略梯度强化学习 （Policy Gradient RL） 算法、近端策略优化 （Proximal Policy Optimization，PPO） 等 

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92GQ8ZnJ97pYdPDSAMWWOSx0iaWHn5CvTpeV87NZWB6hg9CWquFYziao1Q/640?wx_fmt=png)

RLHF，[https://huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)

PPO 算法确定的奖励函数具体计算如下：

+  给定一个输入  ，会生成两个文本 和 ，一个来自于初始的模型，另一个来自于微调的模型。 
+  微调的模型生成的文本还会进入到奖励模型中打分输出 
+  初始模型和微调的模型生成的结果会用 KL 散度约束它们的分布，确保模型不会太偏离原来的模型，并且能输出高质量的回复。 

如果之前没有接触过强化学习，理解这部分原理可能稍微比较困难。没关系，我们可以先简单理解有这么一个过程，只需要大概知道经过强化学习后，LLM 给出的回答会越来越逼近那些在奖励模型中得分比较高的回答。

到目前为止，我们已经基本学习完了制作类似与 ChatGPT 这种 Assistant Model 所需要的原料与菜谱，接下来我们会看看当前业界开源的比较经典的名菜。

### 开源食谱
自从 Meta 3 月初「不小心」泄漏了 LLaMA 的权重之后，开源社区基于 LLaMA 模型涌现了大量创新，在低成本微调、端侧运行、私有化部署以及降低推理成本等展现出巨大潜力，彷佛所有人期待的 LLM Stable Diffusion 时刻就要到来，以至于谷歌工程师在 5 月初发出了 「We Have No Moat， And Neither Does OpenAI」的感叹。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92JPkvACNdztIy2Alp27xibZmuzuyoOX2CuwW2Wlym3jDIw5z1kgqCxOA/640?wx_fmt=png)

LLaMA Family， [https://github.com/RUCAIBox/LLMSurvey](https://github.com/RUCAIBox/LLMSurvey)

这篇文章总结了开源模型在 LLaMA 泄露的 2-3 个月里取得的进展，并以此认为模型的训练没有壁垒，并号召内部拥抱开源、LoRA 微调以及小参数量的模型。这篇文章带来了比较大的争议，也不一定代表着谷歌的官方观点，但我们仍可借此来简单梳理下开源模型的巨大进展。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92B4WstZtd92DicQTY7BtzEQjd97tcHibBBibepVdOblzub6ysGZ5icf04icg/640?wx_fmt=png)

Chatbot Arena Leaderboard，[https://chat.lmsys.org/?leaderboard](https://chat.lmsys.org/?leaderboard)

在 LLaMA 权重泄漏 10 天后，斯坦福大学就发布了 Alpaca 模型，Alpaca-7B 基于 LLaMA-7B 通过 52K Instruction Tuning 指令微调得到。这 52K 的 Instructions 来自于通过 OpenAI 的 text-davinci-003 结合 175 对 self-instrcut 的种子集合，大大简化了指令生成过程，整个指令数据集生成花费不足 500 美元。

![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92hAMJfeK7h4ibmrAxkt42281OXzKJvicERnBbCrCKmlkRjiaSO62DKfAVA/640?wx_fmt=jpeg)

Fine-Tuning Alpaca，[https://crfm.stanford.edu/2023/03/13/alpaca.html](https://crfm.stanford.edu/2023/03/13/alpaca.html)

得到了这个指令遵循数据集后，利用全分片数据并行和混合精度训练等技术，基于 HuggingFace 的训练框架，对 LLaMA 模型进行微调。在首次运行中，在 8 个 80GB A100 显卡上微调 LLaMA-7B 模型，耗时 3 小时，这在大多数云计算供应商上的花费不足 100 美元。

尽管 Alpaca 模型已经大大降低了模型训练的成本，它对于硬件成本要求仍然偏高且训练低效。Alpaca-LoRA 则利用 LoRA 技术，在冻结原模型 LLaMA 参数的情况下，通过往模型中加入额外的网络层，并只训练这些新增的网络层参数。

由于这些新增参数数量较少，这样不仅微调的成本显著下降（使用一块 RTX 4090 显卡，只用 5 个小时就训练了一个与 Alpaca 水平相当的模型，将这类模型对算力的需求降到了消费级），还能获得和全模型微调（full fine-tuning）类似的效果。

LoRA 的核心思想是在原始预训练模型旁边增加一个旁路，先用一个 Linear 层 A，将数据从 d 维降到 r 维，在用第二个 Linear 层 B，将数据从 r 维变回 d 维。LoRA 训练的时候固定预训练模型的参数，只训练降维矩阵 A 和升维矩阵 B。模型的输入输出维度不变，输出时将矩阵 A 和 B 与预训练模型的参数叠加。

![LoRA， Low-Rank Adaptation of Large Language Models](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz926coeQagbibyCQPqvgf05bribNuTWho9LgjdSvria436BFniaFibhKGCGvzA/640?wx_fmt=png)

在推理时，将左右两部分的结果加起来即可（请省略这一句）

在推理时，将左右两部分的结果加起来即可： 。因此，只需要将训练完成的矩阵乘积  跟原来的权重矩阵  合并到一起作为新权重替换原始预训练权重即可。这也正是我们在 Hello World 中所做的事情。

目前 HuggingFace 的 PEFT （Parameter-Efficient Fine-Tuning）提供了模型微调加速的方法，除了这里的 LoRA，还有 Prefix Tuning、P-Tuning、Prompt Tuning 等其他微调加速方法。PEFT 方法通过仅微调少量（额外）模型参数，同时冻结预训练模型中的大部分参数，大大降低了计算与存储成本，在当前的各个垂类微调领域得到了广泛应用。

继 Alpaca 之后，开源社区又很快发布了 Vicuna 模型，据称其中的 13B 模型达到了 ChatGPT 90% 的能力。研究人员从 ShareGPT.com 收集了 70K 个对话，为了确保数据质量，将 HTML 转换回 markdown 并过滤掉一些不合适或低质量的样本。此外，将冗长的对话分成更小的部分，以适应模型的最大上下文长度。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92DIOSlO8eLpDVYuHsic5ichibzb3R6oQYMOncvrYjDOc4g3719WlqKNJcw/640?wx_fmt=png)

Vicuna，[https://lmsys.org/blog/2023-03-30-vicuna](https://lmsys.org/blog/2023-03-30-vicuna)

Vicuna 的训练方法建立在 Alpaca 之上，并进行了以下改进：

+  内存优化：为了使 Vicuna 能够理解长上下文，将最大上下文长度从 alpaca 中的 512 扩展到 2048。还通过 Gradient CheckPointing 和 FlashAttention 来解决内存压力。 
+  多轮对话：调整训练损失，考虑多轮对话，并仅根据聊天机器人的输出进行微调。 
+  通过 Spot 实例降低成本：使用 SkyPilot 托管点来降低成本。该解决方案将 7B 模型的训练成本从 500 美元削减至 140 美元左右，将 13B 模型的训练成本从 1000 美元左右削减至 300 美元。 

为了节省 Vicuna 推理的成本，Vicuna 团队还推出了基于 PagedAttention 的 vLLM 项目，将操作系统的虚拟内存中分页的经典思想引入到 LLM 服务中。

除了上述的几个项目，开源社区仍在不停的向前发展着，此处不再详述，你也可以阅读拾象[关于开源 LLM 的讨论](https://mp.weixin.qq.com/s?__biz=Mzg2OTY0MDk0NQ==&mid=2247502673&idx=1&sn=cecab513962761b949d7ae8996a6b179&scene=21#wechat_redirect)。

## 05 应用领域
经过大量语料数据的预训练和微调之后，LLM 在自己的模型结构中存储了大量的知识，展现出了强大的能力：将原来各自分散的文本分类、机器翻译、情感分析、文本摘要、问答系统等不同 NLP 子领域统一在一起，乃至进一步从 NLP 向外进行领域拓展，将代码生成、图像处理以及多模态等相关任务，典型例子包括微软提出的 Language Models are General-Purpose Interfaces。![Language Models are General-Purpose Interfaces](https://mmbiz.qpic.cn/sz_mmbiz_jpg/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92dIAH7sosNrzicfu3S1bmafhN1YBKNbu7k3xyYxbK7rhvJG27ppswFSw/640?wx_fmt=jpeg)

关于当前最先进的大模型所展现的能力分析，可以参考符尧的这篇文章[6] 和张俊林的这篇文章[7]，其中典型的代表是 In-Context Learning 和 Chain-of-Thought，展现出了令人惊讶的学习能力和推理能力。

![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92a4Jia5LuS2ShIXibpeHiaV14bmjl6EoOqiaoj2UMCawSjxMAzibrYZEYscw/640?wx_fmt=jpeg)

In-Context Learning VS Chain-of-Thought，[https://github.com/RUCAIBox/LLMSurvey](https://github.com/RUCAIBox/LLMSurvey)

尽管目前关于如何商业化落地以 GPT 为代表的大模型技术还存在分歧，业界也都还在探索如何将 LLM 的能力与各自的业务结合，一个很快确定的共识是 LLM 是近年来行业里面难得的突破，各家公司都不想在这次浪潮中掉队。早在 ChatGPT 发布之前，在 22 年中以 Stable Diffusion 为代表的 Foundation Model 开始影响世界，经济学人即开始讨论当前的大模型有可能作为下一代 GPT （General Purpose Techology）。

随着 LLM 技术的迅猛发展，在 GPT-4 发布之后，震惊于 GPT-4 强大能力的人们开始探讨严肃探讨 GPT 对于就业市场的影响，提出 「GPTs are GPTs」，也就是说以 GPT 为代表的大语言模型很有可能是类似于印刷机、蒸汽机、电动机一样可能改变世界的通用技术。

本小节将主要讨论当前仍在快速发展的 AI Native 应用，介绍目前常见的应用框架与不同应用方向，并简单讨论目前非常火热的 Agent Ecosystem。

红杉最近总结了在生成式 AI 领域代表性公司的 AI 50[13]，可以看到除了模型训练、部署、监控等 Infra 层的机会，当前应用层进展也十分迅速，正在形成繁荣的生态。

![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92W72fPUDeiaPVdhf24rXtQ6xjm9NU4InCbckHs3S1PafPOn8ZicwmSqwA/640?wx_fmt=jpeg)

AI 50 in 2023，[https://www.sequoiacap.com/article/ai-50-2023/](https://www.sequoiacap.com/article/ai-50-2023/)

很多公司当前会在某个垂直场景基于自己的专有数据对 Foundation Model 进行 Finetuning，比如医疗健康、法律、教育、金融等场景：

在医疗领域，典型代表是哈工大推出的华驼模型（已经改名为本草），基于医学知识图谱和 GPT3.5 API 构建中文医学指令数据集，并在此基础上对 LLaMA 进行了指令微调。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92TRoyneuiaoGNrbhcRYTgDJ6s38VH453yQX2ogLic15Vy3e1YcbTHEWZw/640?wx_fmt=png)

华驼， 基于中文医学知识的 LLaMA 微调模型

在金融领域，Bloomberg 基于 Bloom 模型和金融领域的数据训练出来了 [BloombergGPT](http://mp.weixin.qq.com/s?__biz=Mzg5Mjc3MjIyMA==&mid=2247561831&idx=1&sn=aa154ab2aa62631e6e318e06688d6090&chksm=c03ab874f74d316250854d698bbf71813fe6e6d38fa8c2e8e12a9a939e18c1f3700311cb99fa&scene=21#wechat_redirect)。Finetuning 相对于从头训练成本小的多，社区也对应有 FinGPT。

![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92cHcbj5XaUQ0vDyOr58QiaHlcJ0IwpBWKNRpTVNVWESxJb52E7BC3ibBA/640?wx_fmt=jpeg)

在法律领域，比较有代表性的是最近很火的 ChatLaw 和 LawGPT。

![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92znV0pdFiaDWDSZJw23MucXEC8haF09JrURUjHMND4icLu529mlP6lpRg/640?wx_fmt=jpeg)

上面展示了几种有代表性的垂类应用，除了需要基于专有领域的数据对 LLM 进行预训练或者微调之外，通常你还会看来类似于 LangChain 这样的 LLM 应用开发框架和类似于 Pinecone 这样的 Vector Database。当前的 LLM Stack 正在迅速演进，下图是红杉对目前的 LLM Stack 的简单总结。

![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92X6COw7l5ibSHibB2npzOhlTPsZT7QwXWYdsfcft6uREfvuaZG0U5GcQg/640?wx_fmt=jpeg)

LLM Stack，[https://www.sequoiacap.com/article/llm-stack-perspective/](https://www.sequoiacap.com/article/llm-stack-perspective/)

当前 Stack 中最火热的项目当属 LangChain，吴恩达和 LangChain 作者有一个课程介绍了 LangChain 的架构与实践，强烈推荐。下图是基于 LangChain 开发的基于文档知识的问答助理，详细展示了当前领域应用开发的典型架构。

+  索引阶段： 
+  将本地的 PDF/Markdown/HTML 这些非结构化的数据源通过 Document Loader 加载并解析出结构化的 Text 
+  通过 Text Splitter 将文档拆分成一个一个的 Chunk 
+  通过 Embedding 将数据存储在 Vector DataBase 中 
+  查询阶段： 
+  用户输入问题后，对问题进行一些预处理，包括安全性验证和拦截等，然后将问题进行 Embedding，得到一个向量表示 
+  通过 Vector DataBase 进行相似度搜索，得到相关文档 
+  将相关文档联合用户问题的内容作为 Prompt 的输入，传给 LLM 从而获得结果 

![LangChain-ChatGLM](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92y8Q3un1DUcXNcFy9ibz9331Zt60T2DBMNIIyRy3LR8q3LoExEfbJ9eQ/640?wx_fmt=png)

OpenAI 最近推出了 Function Calling 的 API，进一步向我们展示了 LLM Application 的巨大潜力。通过前面的介绍我们知道，类似于 GPT4 这样的大语言训练需要很多的资金与时间，随着时间的演进，LLM 内部学习到的信息并不会随着时间同步更新。因此，如果能够解锁 LLM 与外部工具的联动，将确定性的外部工具的能力和 LLM 的自然语言理解与推理的能力结合，则可大幅度扩展当前软件开发的想象空间。

以 Function Calling 为例，如果我们想开发一个天气助手，我们可以通过 LLM 提取出用户问题 What’s the weather like in Boston right now？中对应的 location 的参数，并将其转换成结构化的数据。之后将这个结构化的数据交给公共的 API，则可获得确定性的结果，并最终让 LLM 将这个结构化的结果通过 LLM 封装成人类可以理解的自然语言。

![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz923asM5FGkZlzCZav5L1b5FhN2vWC4J5MKRsiaHaChEaldNicVo7SuoAQQ/640?wx_fmt=jpeg)

OpenAI Function Calling

这即是 Andrej Karpathy 在 2017 年提出的 Software 2.0：基于神经网络的软件设计，真的很有前瞻性了。这进一步引出了当前正在迅速发展的 Agent Ecosystem。AutoGPT ，BabyAGI 和 HuggingGPT 这些项目形象生动地为我们展示了 LLM 的潜力除了在生成内容、故事、论文等方面，它还具有强大的通用问题解决能力。



如果说 ChatGPT 本身的突破体现在人们意识到语言可以成为一种服务，成为人和机器之间最自然的沟通接口，这一轮新发展的关键在于人们意识到语言（不一定是自然语言，也包括命令、代码、错误信息）也是模型和自身、模型和模型以及模型和外部世界之间最自然的接口，让 AI agent 在思考和表达之外增加了调度、结果反馈和自我修正这些新的功能模块。于是在人类用自然语言给 AI 定义任务目标（只有这一步在实质上需要人类参与）之后可以形成一个自动运行的循环：

+  agent 通过语言思考，分解目标为子任务 
+  agent 检讨自己的计划 
+  agent 通过代码语言把子任务分配给别的模型，或者分配给第三方服务，或者分配给自己来执行 
+  agent 观察执行的结果，根据结果思考下一步计划，回到循环开始 

原生的 ChatGPT 让人和 AI 交流成为可能，相当于数学归纳法里 n=0 那一步。而新的 agent ecosystem 实现的是 AI 和自己或者其他 AI 或者外部世界交流，相当于数学归纳法里从 n 推出 n+1 那一步，于是新的维度被展开了。

OpenAI 的 Lilian Weng 最近有一篇博客 「LLM Powered Autonomous Agents」对 Agent 这一领域进行了系统综述，将 Agents 定义为 LLM、Memory、Planing Skills 和 Tool Use 的集合，强烈推荐阅读。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/vHicVZXtcAzCrLurcJWtZX0DiaKh9qPz92r4xUicibrMgpQxqADGHzWjkO1tLa56XetiafKhdDeXZ5SsqniceHf6DYEw/640?wx_fmt=png)

Agent Overview，[https://lilianweng.github.io/posts/2023-06-23-agent/](https://lilianweng.github.io/posts/2023-06-23-agent/)

这或许只是一个新时代。

## 06 期待
本文尝试从程序员的视角去理解 GPT 及其相关技术的最近进展，这个领域涉及到的技术栈比较复杂，不仅仅是模型的结构设计，还有 Infrastructure 的挑战，繁荣发展的应用生态，并且这一领域还在迅速发展着。本文仅仅只是对其粗略的一瞥，还有很多方向并没有提及和探讨：

1.  Alignment 问题，当前的 LLM 的生成内容仍然会存在着胡说八道的 Hallucination 问题，如何将其与实际事实，人类的价值观对齐仍是当前研究的前沿方向 
2.  LLM 的巨大规模对 Infrastructure 提出了新的挑战，如何更高效的训练和推理 LLM 要求人们对 Infrastructure 进行系统性的重构与设计 
3.  当前不同公司和机构都在推出自己的 LLM，如何全面而系统的评估模型的能力也是当前的一个研究方向，我们需要一套统一而权威的 Benchmark 
4.  当前 Agent 生态中一个巨大的限制在于 LLM 的 Context Window 宽度不够，尽管 Anthropic 已经将 Context Window 提升到 100K Tokens，如何让 LLM 更有效率地一次性吞吐上下文仍然是目前的研究热点 
5.  在 LLM 商业化 To B 的领域中，客户数据的隐私问题是目前商业化的一个争议点，如何让客户放心的把数据交给 LLM 厂商，除了当前的一些 「云上专区」的方案，也许大模型和隐私计算的结合也是未来的发展方向之一 
6.  LLM 所展现出的智能让人们对于 Embodied AI 有了更多的想象，试想一下，如果将 Boston Dynamics 机器人强大的机械控制能力和目前 GPT-4 的推理与多模态能力结合，也许科幻小说中的机器人将在不久成为现实 
7.  当前的 LLM 所要求的算力仍然很大，不论是训练还是推理，如果 LLM 将来能够平民化，经历计算机类似于大型机、小型机、微机、个人电脑、智能手机一样的演变，未来的世界也许会更加精彩，或者恐怖 
8.  To Be Continued…… 

在过去的几十年间，程序员这个职业，其中的每个人所代表的内涵与外延经历了沧桑演变：古早时代充满诗意与浪漫的 Ada 伯爵夫人、八十年代个人电脑时代的嬉皮黑客、互联网创业时代改变世界的极客，到后疫情时代经历大规模裁员守在格子间的码农。

当 Github Copilot 已经可以取代你快速实现某个代码模块，当 AutoGPT 已经可以实现你日常中 Google/StackOverflow 的 Debug 流程时，当 ChatGPT 已经帮你快速实现文档总结与书写，当 Stable Diffusion 已经可以帮助你快速实现产品配图时。作为创作者角色的程序员，当如何自处，也许是每个人都应该思考的问题。

最近看到北大微电影《一块石头》，其内容主要在探讨教育如何面对 AI 的冲击与挑战。可以确定的是，我们永远不可能像 AI 读完那么多的书，而且当前 LLM 的推理能力也在迅速提升，我们应该怎么办，也许只有思想在当今这个时代才能更加显示人类的独特性。非常喜欢其中的一句话，既告诉自己，也送给大家。

看过才会留下思想的痕迹，这就是你思考的涟漪。

相关链接：

1.  [https://www.stateof.ai/](https://www.stateof.ai/) 
2.  [https://github.com/RUCAIBox/LLMSurvey](https://github.com/RUCAIBox/LLMSurvey) 
3.  [https://mooc.study.163.com/university/deeplearning_ai#/c](https://mooc.study.163.com/university/deeplearning_ai#/c) 
4.  [https://www.youtube.com/watch?v=bZQun8Y4L2A](https://www.youtube.com/watch?v=bZQun8Y4L2A) 
5. [https://docs.google.com/presentation/d/1EUV7W7X_w0BDrscDhPg7lMGzJCkeaPkGCJ3bN8dluXc/edit](https://docs.google.com/presentation/d/1EUV7W7X_w0BDrscDhPg7lMGzJCkeaPkGCJ3bN8dluXc/edit) 
6.  [https://yaofu.notion.site/514f4e63918749398a1a8a4c660e0d5b](https://yaofu.notion.site/514f4e63918749398a1a8a4c660e0d5b) 
7.  [https://zhuanlan.zhihu.com/p/621438653](https://zhuanlan.zhihu.com/p/621438653) 
8.  [https://github.com/mli/paper-reading](https://github.com/mli/paper-reading) 
9.  [https://huggingface.co/blog/bloom-megatron-deepspeed](https://huggingface.co/blog/bloom-megatron-deepspeed) 
10.  [https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) 
11.  [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053) 
12.  [https://www.sequoiacap.com/article/ai-50-2023/](https://www.sequoiacap.com/article/ai-50-2023/) 

