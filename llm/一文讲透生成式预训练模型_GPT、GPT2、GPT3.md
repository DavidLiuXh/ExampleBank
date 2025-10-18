## GPT
GPT 模型是在 Google BERT 模型之前提出的, 与BERT最大的区别在于GPT采用了传统的语言模型方法进行预训练, 即使用单词的上文来预测单词, 而BERT是采用了双向上下文的信息共同来预测单词。正是因为训练方法上的区别, 使得GPT更擅长处理自然语言生成任务(NLG), 而 BERT 更擅长处理自然语言理解任务(NLU)。

很多时候我们说BERT和GPT的区别在于编码器和解码器，其中编码器可以看到全部而解码器只能看到部分，这个说法其实非常不准确。真正不一样的地方在于预训练任务（以及其对应的目标函数）。GPT的预训练任务是下一个词预测，这比BERT的完形填空难得多，因为它要预测一个开放式的结局。

### 方法动机
聊 GPT，时间就要回溯到2018年，那个时候 NLP 在深度学习上基本还处于 word2vec 以及为不同任务做定制化深度模型的情况，虽然已经有 ELMo 这类预训练模型出现，但是其影响力还远远不足。

当时NLU（Natural Language Understanding）方向存在两大局限：

1.  **有标签的数据相对较少，限制了模型性能的提升。** 
2.  **预训练语言模型存在一定的局限性：** 
+  **不通用：** 不同损失函数在各个任务上表现差异大，训练数据集并没有包含各个NLP任务； 
+  **不统一：** 将预训练语言模型迁移到下游任务的方法不统一，不同的子任务，有时还需要调整模型结构。 

在这个背景下，GPT 第一代预训练语言模型出现了。GPT 是 Generative Pre-Training 的缩写，即**生成式预训练**，来自 OpenAI 的论文 Improving Language Understanding by Generative Pre-Training。Generative Pre-Training 包含两大方面的含义：

+  Pre-Training：指的是大规模自监督预训练，即在大规模没有标签的文本数据上，自监督的完成模型训练，这样就可以利用并发挥出海量无标签文本数据的价值了； 
+  Generative：自监督预训练是通过Next Token Prediction的方式实现的，即是一种生成式的训练方式。 



因此GPT的基本思想是：先在大规模没有标签的数据集上训练一个预训练模型，即generative pre-training的过程；再在子任务小规模有标签的数据集上训练一个微调模型，即discriminative fine-tuning的过程。

### 模型结构
初代GPT选择使用 Transformer 结构，是因为在 NLP 的相关任务中，Transformer 学到的 features 更稳健。与循环神经网络等其他模型相比，Transformer 提供了更结构化的长期记忆，这有利于文本中的长期依赖关系的处理，从而更好的抽取句子层面和段落层面的语义信息。

GPT 利用了任务相关的输入这一表示，将结构化文本输入作为一个连续的 token 序列进行处理。

经典的 Transformer Decoder Block 包含3个子层, 分别是 Masked Multi-Head Attention 层, Encoder-Decoder Attention 层, 以及最后的一个 Feed Forward Network（FFN）全连接层。

**GPT是一个 Decoder-Only 的结构，他根本就没有编码器，自然无需从编码器中获得 Key 和 Value !** 因此，在Decoder-Only 的魔改 Transformer 中，我们往往会取消第二个 Encoder-decoder Attention 子层, 只保留Masked Multi-Head Attention 层, 和 Feed Forward Network 层。

GPT模型结构如下图所示：  
![](https://img-blog.csdnimg.cn/img_convert/ee664ecb495bfc8b0328937056638cf0.png#pic_center)



+  Embedding：词嵌入+位置编码（先前文章已讲过）； 
+  带掩码的多头自注意力机制，让当前时间步和后续时间步的信息的点积在被softmax后都为零（先前文章已讲过）； 
+  输出的向量输入给一个全连接层，而后再输出给下一个Decoder； 
+  GPT有12个Decoder，经过他们后最终的向量经过一个logit的linear层、一个softmax层，就输出了一个下一个词的概率分布函数； 
+  输出的词会作为下一个词的输入。 

### 自监督预训练


![](https://img-blog.csdnimg.cn/direct/449a89a16ca046008ae58e498568e094.png#pic_center)

### 有监督微调
![](https://img-blog.csdnimg.cn/direct/bac8d457e1cd436bb8c5495f08d84cb5.png#pic_center)

### 任务相关的输入和输出变换


![](https://img-blog.csdnimg.cn/img_convert/e9529d8f54f130deeb82a8788ddcd8a6.png#pic_center)

对于GPT处理的4个不同任务，这些任务有的只有一个输入，有的则有多组形式的输入。对于不同的输入，GPT有不同的处理方式，具体方式如下：



+  **分类任务：** 将起始和终止token加入到原始序列两端，输入transformer中得到特征向量，最后经过一个全连接得到预测的概率分布； 
+  **自然语言推理：** 将前提（premise）和假设（hypothesis）通过分隔符（Delimiter）隔开，两端加上起始和终止token。再依次通过transformer和全连接得到预测结果； 
+  **语义相似度：** 输入的两个句子，正向和反向各拼接一次，然后分别输入给transformer，得到的特征向量拼接后再送给全连接得到预测结果； 
+  **问答和常识推理：** 将 个选项的问题抽象化为 个二分类问题，即每个选项分别和内容进行拼接，然后各送入transformer和全连接中，最后选择置信度最高的作为预测结果。 



> 通过这样的方法，这四个自任务就都变成了序列+标注的形式。尽管各自的形式还是稍微有一些不一样，但不管输入形式如何、输出构造如何，中间的Transformer他是不会变的。不管怎样，我们都不去改图中的Transformer的模型结构，这是GPT和之前工作的区别，也是GPT这篇文章的一个核心卖点。
>

### GPT 的更多细节
#### 数据集
GPT使用了BooksCorpus数据集，这个数据集包含 7000 本没有发布的书籍。作者选这个数据集的原因有二：

1.  数据集拥有更长的上下文依赖关系，使得模型能学得更长期的依赖关系； 
2.  这些书籍因为没有发布，所以很难在下游数据集上见到，更能验证模型的泛化能力。 

#### 网络结构
GPT使用了12层的 transformer，使用了掩码自注意力头，掩码的使用使模型看不见未来的信息，得到的模型泛化能力更强。

由于GPT是一个 Decoder-Only 的结构，他没有编码器，自然无需从编码器中获得 Key 和 Value， 因此取消了第二个 Encoder-decoder Attention 子层, 只保留 Masked Multi-Head Attention 层，和 Feed Forward Network 层。

#### 预训练参数
+  使用字节对编码（byte pair encoding，BPE），共有 40,000 个字节对； 
+  词编码的长度为 768； 
+  位置编码也需要学习； 
+  12层的 transformer，每个 transformer 块有 12 个头； 
+  位置编码的长度是 3,072； 
+  Attention，残差，Dropout 等机制用来进行正则化，drop比例为 0.1； 
+  激活函数为 GLEU； 
+  训练的 batchsize 为 64，学习率为 2.5e-4，序列长度为 512，序列 epoch 为 100； 
+  模型参数数量为 1.17 亿。 

#### 有监督微调
+  无监督部分的模型也会用来微调； 
+  训练的epoch为 3，学习率为 6.25e-5，**这表明模型在自监督阶段学到了大量有用的特征。** 

#### GPT 与 BERT
**GPT 和 BERT 的区别**

GPT使用的Transformer的Decoder层（目标函数为标准的语言模型，每次只能看到当前词之前的词，需要用到Decoder中的Masked attention）。

BERT使用的Transformer的Encoder层（目标函数为带[Mask]的语言模型，通过上下文的词预测当前词，对应Encoder）。

#### 为什么初代GPT的性能比BERT差？
GPT预训练时的任务更难（BERT的base就是为了和GPT对比，参数设定几乎一样）；

BERT预训练用的数据集大小几乎是GPT的四倍。

文献和参考  
[1] GPT: Improving Language Understanding by Generative Pre-Training  
[2] [https://zhuanlan.zhihu.com/p/350017443](https://zhuanlan.zhihu.com/p/350017443)  
[3] [https://zhuanlan.zhihu.com/p/609716668](https://zhuanlan.zhihu.com/p/609716668)  
[4] [GPT系列详解：GPT1-GPT2-GPT3](https://zhuanlan.zhihu.com/p/680022511)

## GPT 2
GPT-1作为 GPT 系列的奠基之作，其在预训练、大统一、Decoder-only上做了一系列开创性的工作，但其整体效果并没有超越BERT，只能算是黎明前夜。

时隔不到一年（2019年2月），二代GPT（GPT-2）进化而来，让我们看到了破晓的晨光。

基于 GPT-1 和 BERT 的工作，发现GPT-1这种上下文生成应用面更广，BERT使用编码器和大规模数据集获得了更好的实验效果。一个使用了解码器，一个使用了编码器，换做作者是你，你是否还会继续打回去？GPT-2 的目的就是做这个事情，模型更大，数据更多，效果是否能干掉BERT。

于是作者收集了一个更大的数据集WebText，百万网页数据，同时将模型从1亿参数（110M）变成了15亿参数（1.5B）。但问题是，数据集上去了，模型规模上去了，效果真的能有明显的优势吗？于是作者想到了证明的路子：**Zero-Shot，也是 GPT-2 的核心观点。**

### GPT-2 核心思想
GPT-2的学习目标是使用无监督的预训练模型做有监督的任务。因为文本数据的时序性，一个输出序列可以表示为一系列条件概率的乘积：

上式也可以表示为 ，它的实际意义是根据已知的上文  预测未知的下文 ，因此语言模型可以表示为 。

对于一个有监督的任务，它可以建模为  的形式。

在 decaNLP（The natural language decathlon: Multitask learning as question answering）这篇论文中，作者提出的MQAN模型可以将机器翻译、自然语言推理、语义分析、关系提取等10类任务统一建模为一个分类任务，而无需再为每一个子任务单独设计一个模型。

基于上面的思想，作者认为，当一个语言模型的容量足够大时，它就足以覆盖所有的有监督任务，也就是说**所有的有监督学习都是无监督语言模型的一个子集**。

例如当模型训练完“Micheal Jordan is the best basketball player in the history”语料的语言模型之后，便也学会了(question：“who is the best basketball player in the history ?”，answer:“Micheal Jordan”)的Q&A任务。

综上，GPT-2的核心思想概括为：任何有监督任务都是语言模型的一个子集，当模型的容量非常大且数据量足够丰富时，仅仅靠训练语言模型的学习便可以完成其他有监督学习的任务。

那么有了美好的想法，怎样实现呢？那就**是在大数据和大模型的加持下，由 GPT- 1的 Pre-Training + Fine-Tuning，改成了GPT-2 的 Pre-Training + Prompt Predict （Zero-Shot Learning）。**

### 自监督预训练
训练方式同GPT-1一样，只是数据量变大，模型参数变大，模型结构只是做了几个地方的调整，这些调整更多的是被当作训练时的 trick，而不作为 GPT-2 的创新，我们放到后面再介绍。

### Zero-Shot Predict
下游任务转向做zero-shot而放弃微调（fine-tuning），相较于GPT-1，出现一个新的问题：样本输入的构建不能保持GPT-1的形态，因为模型没有机会学习[Start]，[Delim]，[Extract]这些特殊token。因此，**GPT-2使用一种新的输入形态：增加文本提示，后来被称为prompt**（不是GPT-2第一个提出，他使用的是18年被人提出的方案）。

#### Multitask Learning
GPT-2的论文名称叫《Language Models are Unsupervised Multitask Learners》，Unsupervised 好理解，因为GPT本来就是一个Unsupervised的任务，那么为什么要Multitask Learning，以及什么是Multitask Learning？

现在的语言模型泛化能力比较差，在一个训练集、一个训练任务上训练出来的参数很难直接用到下一个模型里。因此，目前大多数模型都是Narrow Expert，而不是Competent Generalists。**OpenAI希望朝着能够执行许多任务的更通用的系统发展——最终不需要为每个任务手动创建和标记训练数据集。**

**多任务学习的定义：**  
多任务学习（Multi-Task Learning, MTL）是一种机器学习方法，它可以通过同时学习多个相关的任务来提高模型的性能和泛化能力。与单任务学习只针对单个任务进行模型训练不同，多任务学习通过共享模型的部分参数来同时学习多个任务，从而可以更有效地利用数据，提高模型的预测能力和效率。

如何做到多任务学习呢？

**把所有的任务都归结为上下文的问题回答。** 具体的，应该满足如下的条件：

1.  必须对所有任务一视同仁，也就是**喂给模型的数据不能包含这条数据具体是哪个任务**，不能告诉模型这条数据是要做NMT，另一条数据要做情感分类。 
2.  模型也不能包含针对不同任务的特殊模块。给模型输入一段词序列，**模型必须学会自己辨别这段词序列要求的不同的任务类型**，并且进行内部切换，执行不同的操作。 
3.  模型还应具备执行训练任务之外的任务的能力，即 zero shot learning。 

Multitask Question Answering Network, MQAN这篇文章中提出了一个新的在**没有任何特定任务模块或参数**的情况下**联合学习**decaNLP的**所有任务**。把**各种下游子任务都转换为QA任务**，并且让模型通过我们告诉他的自然语言（**Prompt**去自动执行不同的操作，从而完成任务的想法也是GPT-2的关键。这就是为什么提出GPT-2的论文的标题叫《Language Models are Unsupervised Multitask Learners》。

#### Zero-Shot Learning
GPT-2 最大的改变是抛弃了前面“无监督预训练+有监督微调”的模式，而是开创性地引入了 Zero-shot 的技术，即预训练结束后，不需要改变大模型参数即可让它完成各种各样的任务。

**Zero-shot的含义：** 我们用预训练模型做下游任务时，不需要任何额外的标注信息，也不去改模型参数。

#### 从 fine-tune 到 zero-shot
GPT-1中，我们的模型在自然语言上进行预训练，到了给下游任务做微调的时候，我们是引入了很多模型之前从来没有见过的特殊符号，这个符号是针对具体的任务专门设计的，即给GPT的输入进行了特殊的构造，加入了开始符、结束符、分割符。**这些符号，模型要在微调的过程中慢慢认识。**

如果想要做zero-short，即不做任何额外的下游任务训练的话，就没办法让模型去临时学习这些针对特定任务的构造了。因此，我们在构造下游任务的输入的时候，就不能引入特殊的符号，而是要让整个下游任务的输入和之前在预训练的时候看到的文本形式一样。**即要使得输入的形式应该更像自然语言。**

不论是 GPT-1 还是 BERT，NLP 任务中比较主流的 pre-train + fine-tuning 始终还是**需要一定量的下游任务有监督数据去进行额外的训练，在模型层面也需要额外的模块去进行预测，仍然存在较多人工干预的成本**。GPT-2 想彻底解决这个问题，**通过 zero-shot，在迁移到其他任务上的时候不需要额外的标注数据，也不需要额外的模型训练。**

在 GPT-1 中，下游任务需要对不同任务的输入序列进行改造，在序列中加入了开始符、分隔符和结束符之类的特殊标识符，但是在 zero-shot 前提下，我们无法根据不同的下游任务去添加这些标识符，**因为不进行额外的微调训练，模型在预测的时候根本不认识这些特殊标记**。所以**在 zero-shot 的设定下，不同任务的输入序列应该与训练时见到的文本长得一样，也就是以自然语言的形式去作为输入**，例如下面两个任务的输入序列是这样改造的：

> 机器翻译任务：translate to french, { english text }, { french text }  
阅读理解任务：answer the question, { document }, { question }, { answer }
>

为什么上述输入序列的改造是有效的？或者说为什么 zero-shot 是有效的？这里引用原文的一句话：

> Our approach motivates building as large and diverse a dataset as possible in order to collect natural language demonstrations of tasks in as varied of domains and contexts as possible.
>

大概意思是，**从一个尽可能大且多样化的数据集中一定能收集到不同领域不同任务相关的自然语言描述示例**。

例如下图中展示了**英法互译**任务在自然语言中出现的示例，表明了不同任务的任务描述在语料中真实存在：

![](https://img-blog.csdnimg.cn/img_convert/1c36c371d22d3fc4571c3625222cf8ee.png)

所以再次印证 GPT-2 的核心思想就是，**当模型的容量非常大且数据量足够丰富时，仅仅靠语言模型的学习便可以完成其他有监督学习的任务，不需要在下游任务微调。**

#### 从 zero-shot 到 prompting
既然输入的形式也要更像自然语言，那么就应该让模型通过我们的自然语言，去知道现在要去执行什么任务。

要如何做：**实现 Zero-shot learning 的前提就是，我们得能够做到不需要针对下游的任务，不给模型的输入结构做专门的设计；而是只需要给模型指示，也就是提示词（Prompt）就好了。**

为什么prompt能够实现下游任务的预测：

> Our speculation is that a language model with sufficient capacity will begin to learn to infer and perform the tasks demonstrated in natural language sequences in order to better predict them, regardless of their method of procurement. ——OpenAI
>

大概意思是，**当数据量足够大、模型能力足够强的时候，语言模型会学会推测并执行用自然语言提出的任务，因为这样可以更好的实现下一词预测。**(In-context Learning的雏形)

### 模型结构
在模型结构方面，整个 GPT-2 的模型框架与 GPT-1 相同，只是做了几个地方的调整，这些调整更多的是被当作训练时的 trick，而不作为 GPT-2 的创新，具体为以下几点：

1.  后置层归一化（ post-norm ）改为前置层归一化（ pre-norm ）; 
2.  在模型最后一个自注意力层之后，额外增加一个层归一化; 
3.  调整参数的初始化方式，按残差层个数进行缩放，缩放比例为 1 : sqrt(n); 
4.  输入序列的最大长度从 512 扩充到 1024; 
5.  模型层数扩大，从 GPT-1 的 12 层最大扩大到 48 层，参数量也从 1 亿扩大到 15 亿。 



其中，关于 post-norm 和 pre-norm 可以参考《Learning Deep Transformer Models for Machine Translation》。两者的主要区别在于，post-norm 将 transformer 中每一个 block 的层归一化放在了残差层之后，而 pre-norm 将层归一化放在了每个 block 的输入位置，如下图所示：

![](https://img-blog.csdnimg.cn/img_convert/6321ea4ddd0ca33c46c9c4d03f2096f0.png)

GPT-2 进行上述模型调整的主要原因在于，**随着模型层数不断增加，梯度消失和梯度爆炸的风险越来越大，这些调整能够减少预训练过程中各层之间的方差变化，使梯度更加稳定。**

最终 GPT-2 提供了四种规模的模型：

![](https://img-blog.csdnimg.cn/img_convert/18d5538b6752c4e0ebe05feac67e543e.png)

其中 117M 参数等价于 GPT-1 模型，345M 参数模型用于对标同期的 BERT-large 模型。

### 训练数据
在训练数据方面，为了保证 zero-shot 的效果，必须要足够大且覆盖面广。所以 GPT-2 专门爬取了大量的网络文本数据，最后得到的数据集叫 WebText。它选取了 Reddit 上的高质量帖子，最终得到 4500w 网页链接，800w 有效的文本文档，语料大小为 40G。

### 模型效果
在实验效果上，由于 GPT-2 主要是做 zero-shot，所以在实验部分，很多的实验对比都是在无监督的设定下进行的，也就是说他对比的都是无监督的算法。

+  在 8 个语言模型任务中，仅仅通过 zero-shot 学习，GPT-2 就有7个超过了 state-of-the-art 的方法； 
+  在 “Children's Book Test” 数据集上的命名实体识别任务中，GPT-2 超过了 state-of-the-art 的方法约7%； 
+  “LAMBADA” 是测试模型捕捉长期依赖的能力的数据集，GPT-2 将困惑度从 99.8 降到了 8.6； 
+  在阅读理解数据中，GPT-2 超过了 4 个baseline模型中的 3 个； 
+  在法译英任务中，GPT-2 在 zero-shot 学习的基础上，超过了大多数的无监督方法，但是比有监督的 state-of-the-art模型要差； 
+  GPT-2 在文本总结的表现不理想，但是它的效果也和有监督的模型非常接近。 

从上述效果可以看到，GPT-2 在一些任务上与有监督微调的方法相比还是有一些差距的，这可能也是 GPT-2 在当时影响力没有那么大的一个原因。但 **GPT-2 在较多任务上对比无监督算法取得了一定的提升，证明了 zero-shot 的能力，这也成为了GPT系列的破晓晨光。**

**文献和参考：**

[1] GPT-2: Language Models are Unsupervised Multitask Learners  
[2] [https://zhuanlan.zhihu.com/p/350017443](https://zhuanlan.zhihu.com/p/350017443)  
[3] [https://zhuanlan.zhihu.com/p/609716668](https://zhuanlan.zhihu.com/p/609716668)  
[4] [https://zhuanlan.zhihu.com/p/680022511](https://zhuanlan.zhihu.com/p/680022511)

## GPT 3
GPT-1 模型指出，如果用 Transformer 的解码器和大量的无标签样本去预训练一个语言模型，然后在子任务上提供少量的标注样本做 **Fine-Tune**，就可以很大的提高模型的性能。

GPT-2 则是更往前走了一步，说在子任务上不去提供任何相关的训练样本，而是直接用足够大的预训练模型去理解自然语言表达的要求，并基于此做预测，因此主推了 **Zero-Shot**，虽然具有较高的创新度和吸引力，但其效果上还是不够惊艳，所以在业界也并没有取得比较大的影响力。

为了解决效果上的问题，时隔一年多，GPT-3 以势不可挡之势卷土重来。GPT-3 不再去追求那种极致的不需要任何样本就可以表现很好的模型，而是考虑像人类的学习方式那样，仅仅使用极少数样本就可以掌握某一个任务，因此就引出了 GPT-3 标题 Language Models are Few-Shot Learners。

**GPT-3 中的 few-shot learning，只是在预测是时候给几个例子，并不微调网络。** GPT-2用 zero-shot 去讲了 multitask Learning 的故事，GPT-3 使用 meta-learning 和 in-context learning 去讲故事。

### 自监督预训练
训练方式同 GPT-1 和 GPT-2 一样，只是数据量和模型参数都得到了巨幅提升，网络结构也做了一些优化，但这不是 GPT-3 的重点，我们放到后面讲解。

### In-context learning
**In-context learning 是 GPT-3 运用的一个重要概念，本质上是属于 few-shot learning，只不过这里的 few-shot 只是在预测是时候给几个例子，并不微调网络，即不会再更新模型权重。**

要理解 in-context learning，我们需要先理解 meta-learning（元学习）。对于一个少样本的任务来说，模型的初始化值非常重要，从一个好的初始化值作为起点，模型能够尽快收敛，使得到的结果非常快的逼近全局最优解。Meta-learning 的核心思想在于通过少量的数据寻找一个合适的初始化范围，使得模型能够在有限的数据集上快速拟合，并获得不错的效果。

我们使用MAML（Model-Agnostic Meta-Learning）算法来理解一下 Meta-learning。正常的监督学习是将一个批次的数据打包成一个batch进行学习，但是元学习是将一个个任务打包成batch，每个batch分为支持集（support set）和质询集（query set），类似于学习任务中的训练集和测试集。



![](https://img-blog.csdnimg.cn/img_convert/40cfa4cdb15c75a73054c3b602ebdc0c.png#pic_center)



对一个网络模型 ，其参数表示为 ，它的初始化值被叫做meta-initialization。MAML的目标则是学习一组meta-initialization，能够快速应用到其它任务中。MAML的迭代涉及两次参数更新，分别是内循环（inner loop）和外循环（outer loop）。内循环是根据任务标签快速的对具体的任务进行学习和适应，而外学习则是对meta-initialization进行更新。直观的理解，我用一组meta-initialization去学习多个任务，如果每个任务都学得比较好，则说明这组meta-initialization是一个不错的初始化值，否则我们就去对这组值进行更新。

**GPT-3 中据介绍的 in-context learning（情境学习）就是元学习的内循环，而基于语言模型的SGD则是外循环**，如下图所示：

![](https://img-blog.csdnimg.cn/img_convert/6f2c8d0af2789e9e15d103d90f631150.png#pic_center)



GPT-3 的 few-shot learning 是不会做梯度下降，它是怎么做的？



**只做预测，不做训练。**我们希望 Transformer 在做前向推理的时候，能够通过注意力机制，从我们给他的输入之中抽取出有用的信息，从而进行预测任务，而预测出来的结果其实也就是我们的任务指示了。**这就是上下文学习（In context learning）。**

### GPT-3 的上下文学习能力究竟从何而来
![](https://img-blog.csdnimg.cn/img_convert/6f472baaf3feaa0405fbc1ffdbcdb788.png#pic_center)

如上图所示，GPT-3 在下游任务的评估与预测时，提供了三种不同的方法：



+  Zero-shot：仅使用当前任务的自然语言描述，不进行任何梯度更新； 
+  One-shot：当前任务的自然语言描述，加上一个简单的输入输出样例，不进行任何梯度更新； 
+  Few-shot：当前任务的自然语言描述，加上几个简单的输入输出样例，不进行任何梯度更新。 

其中 Few-shot 也被称为 in-context learning，虽然它与 fine-tuning 一样都需要一些有监督标注数据，但是两者的区别是：

1.  fine-tuning 基于标注数据对模型参数进行更新，而 in-context learning 使用标注数据时不做任何的梯度回传，模型参数不更新 **【本质区别】**； 
2.  in-context learning 依赖的数据量（10～100）远远小于 fine-tuning 一般的数据量。 

最终通过大量下游任务实验验证，Few-shot 效果最佳，One-shot 效果次之，Zero-shot 效果最差，如下图所示。图中横坐标为模型参数量，纵坐标为任务精度，图中大量灰色线表示不同下游任务，橙色/绿色/蓝色线是下游任务效果的平均值。



![](https://img-blog.csdnimg.cn/img_convert/6e7f950335b09a4cc7d0180cb0025d36.png#pic_center)

当然这个模式也有缺陷：

1.  GPT-3 的输入窗口长度是有限的，不可能无限的堆叠example的数量，即有限的输入窗口限制了我们利用海量数据的能力； 
2.  每次做一次新的预测，模型都要从输入的中间抓取有用的信息。可是我们做不到把从上一个输入中抓取到的信息存起来，存在模型中，用到下一次输入里。 

### 模型结构
GPT-3 沿用了GPT-2 的结构，但是在网络容量上做了很大的提升，并且使用了一个 Sparse Transformer 的架构，具体如下：

1.  GPT-3采用了96层的多头transformer，头的个数为 96； 
2.  词向量的长度是12,888； 
3.  上下文划窗的窗口大小提升至2,048个token； 
4.  使用了alternating dense和locally banded sparse attention。 

sparse attention 与传统 self-attention（称为 dense attention） 的区别在于：

+  dense attention：每个 token 之间两两计算 attention，复杂度 O(n²) 
+  sparse attention：每个 token 只与其他 token 的一个子集计算 attention，复杂度 O(n*logn) 

具体来说，sparse attention 除了相对距离不超过 k 以及相对距离为 k，2k，3k，... 的 token，其他所有 token 的注意力都设为 0，如下图所示：

![](https://img-blog.csdnimg.cn/direct/4d89632ffd1140d79a5da60d1a6dcfa2.png#pic_center)

使用 sparse attention 的好处主要有以下两点：

1.  减少注意力层的计算复杂度，节约显存和耗时，从而能够处理更长的输入序列； 
2.  具有“局部紧密相关和远程稀疏相关”的特性，对于距离较近的上下文关注更多，对于距离较远的上下文关注较少。 

> 关于 sparse attention 详情可参考《Generating Long Sequences with Sparse Transformers》。
>

最终 GPT-3 在训练过程中得到了如下不同规模的模型,其中规模最大的模型称为 GPT-3，模型参数量为 1750 亿。

![](https://img-blog.csdnimg.cn/img_convert/70df3477673a9189ce2c2bba2177c2b5.png#pic_center)

1750 亿参数在当时是什么概念？大家可以通过下面这张对比图来直观的感受下：

![](https://img-blog.csdnimg.cn/img_convert/27c3c2036beaa55ea99f752379abde63.png#pic_center)

### 训练数据
由于 GPT-3 在模型规模上的扩大，在训练数据方面也必须进行扩充来适配更大的模型使其发挥出相应的能力。

GPT-3 使用了多个数据集，其中最大的是 CommonCrawl，原始未处理的数据达到了 45TB，其实在 GPT-2 的时候他们就有考虑使用这个数据集，但是后来还是觉得这个数据集太脏了所以没用，但是现在 GPT-3 的模型规模太大了，使得训练对数据量的需求也增加了很多，他们不得不重新考虑这个数据集。因此，他们必须在这个数据集上做一些额外的数据清洗工作来尽量保证数据的质量。

数据处理主要包括以下几个部分：

1.  使用高质量数据作为正例，训练 LR 分类算法，对 CommonCrawl 的所有文档做初步过滤； 
2.  利用公开的算法做文档去重，减少冗余数据； 
3.  加入已知的高质量数据集。 

其中“高质量数据”主要是指 BERT、GPT、GPT-2 使用过的数据，最终处理完成后使用的数据规模约 570G。

![](https://img-blog.csdnimg.cn/img_convert/95d4f0771be1f36712652101a6ca18ee.png#pic_center)

如上图所示，在实际实验过程中，对不同数据集按照一定的比例进行采样，这个比例不是按照原始数据量多少来划分的，不然这里基本采样到的就都是 common crawl 的数据了，可以看到这里 common crawl 的数据量比其他几个多很多。进行采样的原因主要考虑到，就算做了一些数据清洗还是觉得 common crawl 的数据质量不如其他几个。最终采样的时候，虽然 common crawl 的数据量是其他几个数据集的上百倍，但是实际占比是 60%，有 40% 的数据是能够保证质量的。

### 实验分析
GPT-3 花了大部分篇幅介绍了各种 NLP 任务上的实验结果和分析，大家如果对某个任务感兴趣的话可以自行阅读一下论文对应的章节，本文就不做详细介绍了。

下图是 GPT-3 的一个重要分析结果：

![](https://img-blog.csdnimg.cn/img_convert/418c7546b749c0923d013db0277adb87.png#pic_center)

图中横坐标为计算量，可以简单理解为模型规模或者数据量（不止如此），纵坐标为任务精度。可以看到，**当我们想要线性的提升一个任务的效果时，往往需要指数级的提升模型的规模和所需的数据量。**这也成为了大模型后续持续爆发的指引。

### GPT-3 的性能
仅仅用惊艳很难描述GPT-3的优秀表现。首先，在大量的语言模型数据集中，GPT-3 超过了绝大多数的 zero-shot 或者 few-shot 的 state-of-the-art 方法。另外 GPT-3 在很多复杂的NLP任务中也超过了 fine-tune 之后的 state-of-the-art 方法，例如闭卷问答，模式解析，机器翻译等。除了这些传统的 NLP 任务，GPT-3 在一些其他的领域也取得了非常震惊的效果，例如进行数学加法，文章生成，编写代码等。

**GPT-3 通过“大力出奇迹”表现出势不可挡的性能和潜力，后续有研究者将其称为涌现（Emergent），从此人们进入了痴迷于“大AI”的时代。**

![](https://img-blog.csdnimg.cn/img_convert/47ffac7b3b5cdb1a8681733144035cda.png#pic_center)

> 上图来自论文 Are Emergent Abilities of Large Language Models a Mirage?
>

当然 GPT-3 也还有很多问题，这就是 GPT-3.5、GPT-4、GPT-5 要解决的问题了。这里引用 Sam Altman 本人的话来给 GPT-3 做个总结：

> The GPT-3 hype is way too much. It’s impressive (thanks for the nice compliments!) but it still has serious weaknesses and sometimes makes very silly mistakes. AI is going to change the world, but GPT-3 is just a very early glimpse. We have a lot still to figure out.
>
> GPT-3 的炒作太过分了。 它令人印象深刻（感谢您的赞美！），但它仍然有严重的弱点，有时会犯非常愚蠢的错误。 人工智能将改变世界，但 GPT-3 只是一个非常早期的雏形。 我们还有很多事情需要弄清楚。
>

**文献和参考：**  
[1] GPT-3: Language Models are Few-Shot Learners  
[2] [https://zhuanlan.zhihu.com/p/350017443](https://zhuanlan.zhihu.com/p/350017443)  
[3] [https://zhuanlan.zhihu.com/p/609716668](https://zhuanlan.zhihu.com/p/609716668)  
[4] [https://zhuanlan.zhihu.com/p/680022511](https://zhuanlan.zhihu.com/p/680022511)

