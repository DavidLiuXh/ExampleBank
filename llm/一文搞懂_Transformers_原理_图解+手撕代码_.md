Transformers 亮相以来彻底改变了深度学习模型。

今天，我们将揭示 Transformers 背后的核心概念：注意力机制、编码器-解码器架构、多头注意力等等。

  
![](https://files.mdnice.com/user/3721/6027fce1-6f16-49f7-8157-4bdbec8ec15d.png)  


通过 Python 代码片段，让你深入了解其原理。

## 一、理解注意力机制
注意力机制是神经网络中一个迷人的概念，特别是在涉及到像 NLP 这样的任务时。它就像给模型一个聚光灯，让它能够集中注意力在输入序列的某些部分，同时忽略其他部分，就像我们人类在理解句子时关注特定的单词或短语一样。

现在，让我们深入了解一种特定类型的注意力机制，称为自注意力，也称为内部注意力。想象一下，当你阅读一句话时，你的大脑会自动突出显示重要的单词或短语来理解意思。这就是神经网络中自注意力的基本原理。它使序列中的每个单词都能“关注”其他单词，包括自己在内，以更好地理解上下文。

## 二、自注意力是如何工作的？
以下是自注意力在一个简单示例中的工作原理：

考虑一句话：“The cat sat on the mat.”

**嵌入**

首先，模型将输入序列中的每个单词嵌入到一个高维向量表示中。这个嵌入过程允许模型捕捉单词之间的语义相似性。

**查询、键和值向量**

接下来，模型为序列中的每个单词计算三个向量：查询向量、键向量和值向量。在训练过程中，模型学习这些向量，每个向量都有不同的作用。查询向量表示单词的查询，即模型在序列中寻找的内容。键向量表示单词的键，即序列中其他单词应该注意的内容。值向量表示单词的值，即单词对输出所贡献的信息。

**注意力分数**

一旦模型计算了每个单词的查询、键和值向量，它就会为序列中的每一对单词计算注意力分数。这通常通过取查询向量和键向量的点积来实现，以评估单词之间的相似性。

**SoftMax 归一化**

然后，使用 softmax 函数对注意力分数进行归一化，以获得注意力权重。这些权重表示每个单词应该关注序列中其他单词的程度。注意力权重较高的单词被认为对正在执行的任务更为关键。

**加权求和**

最后，使用注意力权重计算值向量的加权和。这产生了每个序列中单词的自注意力机制输出，捕获了来自其他单词的上下文信息。



![](https://files.mdnice.com/user/3721/5aa5ff7b-d060-4341-9feb-8262e7c25d0a.png)



下面是一个计算注意力分数的简单解释：

```python
# 安装 PyTorch
!pip install torch==2.2.1+cu121

# 导入库
import torch
import torch.nn.functional as F

# 示例输入序列
input_sequence = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

# 生成 Key、Query 和 Value 矩阵的随机权重
random_weights_key = torch.randn(input_sequence.size(-1), input_sequence.size(-1))
random_weights_query = torch.randn(input_sequence.size(-1), input_sequence.size(-1))
random_weights_value = torch.randn(input_sequence.size(-1), input_sequence.size(-1))

# 计算 Key、Query 和 Value 矩阵
key = torch.matmul(input_sequence, random_weights_key)
query = torch.matmul(input_sequence, random_weights_query)
value = torch.matmul(input_sequence, random_weights_value)

# 计算注意力分数
attention_scores = torch.matmul(query, key.T) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))

# 使用 softmax 函数获得注意力权重
attention_weights = F.softmax(attention_scores, dim=-1)

# 计算 Value 向量的加权和
output = torch.matmul(attention_weights, value)

print("自注意力机制后的输出:")
print(output)
```

## 三、Transformer 模型的基础
在我们深入探讨Transformer模型的复杂工作原理之前，让我们花点时间欣赏其开创性的架构。正如我们之前讨论的，Transformer模型通过引入围绕自注意力机制的新颖方法，重塑了自然语言处理（NLP）的格局。在接下来的章节中，我们将揭开Transformer模型的核心组件，阐明其编码器-解码器架构、位置编码、多头注意力和前馈网络。

**编码器-解码器架构**

在Transformer的核心是其编码器-解码器架构——两个关键组件之间的共生关系，分别负责处理输入序列和生成输出序列。编码器和解码器中的每一层都包含相同的子层，包括自注意力机制和前馈网络。这种架构不仅有助于全面理解输入序列，而且能够生成上下文丰富的输出序列。

**位置编码**

尽管Transformer模型具有强大的功能，但它缺乏对元素顺序的内在理解——这是位置编码所解决的一个缺点。通过将输入嵌入与位置信息结合起来，位置编码使模型能够区分序列中元素的相对位置。这种细致的理解对于捕捉语言的时间动态和促进准确理解至关重要。

**多头注意力**

Transformer模型的一个显著特征是它能够同时关注输入序列的不同部分——这是多头注意力实现的。通过将查询、键和值向量分成多个头，并进行独立的自注意力计算，模型获得了对输入序列的细致透视，丰富了其表示，带有多样化的上下文信息。

**前馈网络**

与人类大脑能够并行处理信息的能力类似，Transformer模型中的每一层都包含一个前馈网络——一种能够捕捉序列中元素之间复杂关系的多功能组件。通过使用线性变换和非线性激活函数，前馈网络使模型能够在语言的复杂语义景观中航行，促进文本的稳健理解和生成。

## 四、Transformer 组件的详细说明
要实现，首先运行位置编码、多头注意力机制和前馈网络的代码，然后是编码器、解码器和Transformer架构。

```python
#import libraries
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```



**1、位置编码**

在Transformer模型中，位置编码是一个关键组件，它将关于标记位置的信息注入到输入嵌入中。

与循环神经网络（RNNs）或卷积神经网络（CNNs）不同，由于其置换不变性，Transformers 缺乏对标记位置的内在知识。位置编码通过为模型提供位置信息来解决这一限制，使其能够按照正确的顺序处理序列。

**位置编码的概念**

通常在将输入嵌入传入Transformer模型之前，会将位置编码添加到嵌入中。它由一组具有不同频率和相位的正弦函数组成，允许模型根据它们在序列中的位置区分标记。

**位置编码的公式如下**

假设您有一个长度为L的输入序列，并且需要在该序列中找到第k个对象的位置。位置编码由不同频率的正弦和余弦函数给出：

![](https://files.mdnice.com/user/3721/43c78eeb-d060-497c-93e2-57a4be005986.png)



其中：

+ k：输入序列中对象的位置，0≤k<L/2
+ d：输出嵌入空间的维度
+ P(k,j)：位置函数，用于将输入序列中的位置k映射到位置矩阵的索引(k,j)
+ n：用户定义的标量，由《Attention Is All You Need》的作者设置为10,000。
+ i：用于将列索引映射到0≤i<d/2的值，单个i值同时映射到正弦和余弦函数。



**不同的位置编码方案**

在Transformer中使用了各种位置编码方案，每种方案都有其优点和缺点：

+ 固定位置编码：在这种方案中，位置编码是预定义的，并对所有序列固定不变。虽然简单高效，但固定位置编码可能无法捕捉序列中的复杂模式。
+ 学习位置编码：另一种选择是在训练过程中学习位置编码，使模型能够自适应地从数据中捕捉位置信息。学习位置编码提供了更大的灵活性，但需要更多的参数和计算资源。

**位置编码的实现**

让我们用Python实现位置编码：

```plain
# 位置编码的实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(
        torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + x + self.pe[:, :x.size(1)]
        return x

# 示例用法
d_model = 512
max_len = 100
num_heads = 8

# 位置编码
pos_encoder = PositionalEncoding(d_model, max_len)

# 示例输入序列
input_sequence = torch.randn(5, max_len, d_model)

# 应用位置编码
input_sequence = pos_encoder(input_sequence)
print("输入序列的位置编码:")
print(input_sequence.shape)
```



**2、多头注意力机制**

在Transformer架构中，多头注意力机制是一个关键组件，它使模型能够同时关注输入序列的不同部分。它允许模型捕捉序列内的复杂依赖关系和关联，从而提高了语言翻译、文本生成和情感分析等任务的性能。

![](https://files.mdnice.com/user/3721/bd34dc35-257c-4fc0-a094-b0160b95052d.png)  
**多头注意力的重要性**

多头注意力机制具有几个优点：

+ 并行化：通过同时关注输入序列的不同部分，多头注意力显著加快了计算速度，使其比传统的注意力机制更加高效。
+ 增强表示：每个注意力头都关注输入序列的不同方面，使模型能够捕捉各种模式和关系。这导致输入的表示更丰富、更强大，增强了模型理解和生成文本的能力。
+ 改进泛化性：多头注意力使模型能够关注序列内的局部和全局依赖关系，从而提高了跨不同任务和领域的泛化性。

**多头注意力的计算：**

让我们分解计算多头注意力所涉及的步骤：

+ 线性变换：输入序列经历可学习的线性变换，将其投影到多个较低维度的表示，称为“头”。每个头关注输入的不同方面，使模型能够捕捉各种模式。
+ 缩放点积注意力：每个头独立地计算输入序列的查询、键和值表示之间的注意力分数。这一步涉及计算令牌及其上下文之间的相似度，乘以模型深度的平方根进行缩放。得到的注意力权重突出了每个令牌相对于其他令牌的重要性。
+ 连接和线性投影：来自所有头的注意力输出被连接并线性投影回原始维度。这个过程将来自多个头的见解结合起来，增强了模型理解序列内复杂关系的能力。

**用代码实现**

让我们将理论转化为代码：

```python
# 多头注意力的代码实现
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        # 查询、键和值的线性投影
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        # 输出线性投影
        self.output_linear = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
      batch_size, seq_length, d_model = x.size()
      return x.view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        
        # 线性投影
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # 分割头部
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        # 缩放点积注意力
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.depth)
        
        # 如果提供了掩码，则应用掩码
        if mask is not None:
            scores += scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重并应用softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 应用注意力到值
        attention_output = torch.matmul(attention_weights, value)
        
        # 合并头部
        batch_size, _, seq_length, d_k = attention_output.size()
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size,
        seq_length, self.d_model)
        
        # 线性投影
        attention_output = self.output_linear(attention_output)
        
        return attention_output

# 示例用法
d_model = 512
max_len = 100
num_heads = 8
d_ff = 2048

# 多头注意力
multihead_attn = MultiHeadAttention(d_model, num_heads)

# 示例输入序列
input_sequence = torch.randn(5, max_len, d_model)

# 多头注意力
attention_output= multihead_attn(input_sequence, input_sequence, input_sequence)
print("attention_output shape:", attention_output.shape)
```

**3、前馈网络**

在Transformer的背景下，前馈网络在处理信息和从输入序列中提取特征方面发挥着关键作用。它们是模型的支柱，促进了不同层之间表示的转换。

**前馈网络的作用**

每个Transformer层内的前馈网络负责对输入表示应用非线性变换。它使模型能够捕捉数据中的复杂模式和关系，促进了高级特征的学习。

**前馈层的结构和功能**

前馈层由两个线性变换组成，两者之间通过一个非线性激活函数（通常是ReLU）分隔。让我们来解析一下结构和功能：

+ 线性变换1：使用可学习的权重矩阵将输入表示投影到更高维度的空间中。
+ 非线性激活：第一个线性变换的输出通过非线性激活函数（例如ReLU）传递。这引入了模型的非线性，使其能够捕捉数据中的复杂模式和关系。
+ 线性变换2：激活函数的输出然后通过另一个可学习的权重矩阵投影回原始的维度空间中。

**用代码实现**

让我们在Python中实现前馈网络：

```python
# 前馈网络的代码实现
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 线性变换1
        x = self.relu(self.linear1(x))
        
        # 线性变换2
        x = self.linear2(x)
        
        return x

# 示例用法
d_model = 512
max_len = 100
num_heads = 8
d_ff = 2048

# 多头注意力
multihead_attn = MultiHeadAttention(d_model, num_heads)

# 前馈网络
ff_network = FeedForward(d_model, d_ff)

# 示例输入序列
input_sequence = torch.randn(5, max_len, d_model)

# 多头注意力
attention_output= multihead_attn(input_sequence, input_sequence, input_sequence)

# 前馈网络
output_ff = ff_network(attention_output)
print('input_sequence',input_sequence.shape)
print("output_ff", output_ff.shape)
```

**4、编码器**

在Transformer模型中起着至关重要的作用，其主要任务是将输入序列转换为有意义的表示，捕捉输入的重要信息。

![](https://files.mdnice.com/user/3721/d5bb3f50-9b50-43a6-ab3d-7ba6c168f288.png)  
每个编码器层的结构和功能

编码器由多个层组成，每个层依次包含以下组件：输入嵌入、位置编码、多头自注意力机制和位置逐点前馈网络。

1.  输入嵌入：我们首先将输入序列转换为密集向量表示，称为输入嵌入。我们使用预训练的词嵌入或在训练过程中学习的嵌入，将输入序列中的每个单词映射到高维向量空间中。 
2.  位置编码：我们将位置编码添加到输入嵌入中，以将输入序列的顺序信息合并到其中。这使得模型能够区分序列中单词的位置，克服了传统神经网络中缺乏顺序信息的问题。 
3.  多头自注意力机制：在位置编码之后，输入嵌入通过一个多头自注意力机制。这个机制使编码器能够根据单词之间的关系权衡输入序列中不同单词的重要性。通过关注输入序列的相关部分，编码器可以捕捉长距离的依赖关系和语义关系。 
4.  位置逐点前馈网络：在自注意力机制之后，编码器对每个位置独立地应用位置逐点前馈网络。这个网络由两个线性变换组成，两者之间通过一个非线性激活函数（通常是ReLU）分隔。它有助于捕捉输入序列中的复杂模式和关系。 

**代码实现**

让我们来看一下用Python实现带有输入嵌入和位置编码的编码器层的代码：



```python
# 编码器的代码实现
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        
        # 自注意力层
        attention_output= self.self_attention(x, x,
        x, mask)
        attention_output = self.dropout(attention_output)
        x = x + attention_output
        x = self.norm1(x)
        
        # 前馈层
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout(feed_forward_output)
        x = x + feed_forward_output
        x = self.norm2(x)
        
        return x

d_model = 512
max_len = 100
num_heads = 8
d_ff = 2048


# 多头注意力
encoder_layer = EncoderLayer(d_model, num_heads, d_ff, 0.1)

# 示例输入序列
input_sequence = torch.randn(1, max_len, d_model)

# 多头注意力
encoder_output= encoder_layer(input_sequence, None)
print("encoder output shape:", encoder_output.shape)
```

**5、解码器**

在Transformer模型中，解码器在基于输入序列的编码表示生成输出序列方面起着至关重要的作用。它接收来自编码器的编码输入序列，并将其用于生成最终的输出序列。  
![](https://files.mdnice.com/user/3721/24cf2b6f-6f29-4829-9bd8-091dce17c029.png)

**解码器的功能**

解码器的主要功能是生成输出序列，同时注意到输入序列的相关部分和先前生成的标记。它利用输入序列的编码表示来理解上下文，并对生成下一个标记做出明智的决策。

**解码器层及其组件**

解码器层包括以下组件：

1.  输出嵌入右移：在处理输入序列之前，模型将输出嵌入向右移动一个位置。这确保解码器中的每个标记在训练期间都能从先前生成的标记接收到正确的上下文。 
2.  位置编码：与编码器类似，模型将位置编码添加到输出嵌入中，以合并标记的顺序信息。这种编码帮助解码器根据标记在序列中的位置进行区分。 
3.  掩码的多头自注意力机制：解码器采用掩码的多头自注意力机制，以便注意输入序列的相关部分和先前生成的标记。在训练期间，模型应用掩码以防止注意到未来的标记，确保每个标记只能注意到前面的标记。 
4.  编码器-解码器注意力机制：除了掩码的自注意力机制外，解码器还包括编码器-解码器注意力机制。这种机制使解码器能够注意到输入序列的相关部分，有助于生成受输入上下文影响的输出标记。 
5.  位置逐点前馈网络：在注意力机制之后，解码器对每个标记独立地应用位置逐点前馈网络。这个网络捕捉输入和先前生成的标记中的复杂模式和关系，有助于生成准确的输出序列。 

**使用代码实现**

```python
# 解码器的代码实现
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        
        # 掩码的自注意力层
        self_attention_output= self.masked_self_attention(x, x, x, tgt_mask)
        self_attention_output = self.dropout(self_attention_output)
        x = x + self_attention_output
        x = self.norm1(x)
        
        # 编码器-解码器注意力层
        enc_dec_attention_output= self.enc_dec_attention(x, encoder_output, 
        encoder_output, src_mask)
        enc_dec_attention_output = self.dropout(enc_dec_attention_output)
        x = x + enc_dec_attention_output
        x = self.norm2(x)
        
        # 前馈层
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout(feed_forward_output)
        x = x + feed_forward_output
        x = self.norm3(x)
        
        return x

# 定义DecoderLayer的参数
d_model = 512  # 模型的维度
num_heads = 8  # 注意力头的数量
d_ff = 2048    # 前馈网络的维度
dropout = 0.1  # 丢弃概率
batch_size = 1 # 批量大小
max_len = 100  # 序列的最大长度

# 定义DecoderLayer实例
decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)


src_mask = torch.rand(batch_size, max_len, max_len) > 0.5
tgt_mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0) == 0

# 将输入张量传递到DecoderLayer
output = decoder_layer(input_sequence, encoder_output, src_mask, tgt_mask)

# 输出形状
print("Output shape:", output.shape)
```

## 五、Transformer 模型架构
前几节讨论的各种组件的综合体。让我们将编码器、解码器、注意力机制、位置编码和前馈网络的知识汇集起来，以了解完整的 Transformer 模型是如何构建和运作的。  
![](https://files.mdnice.com/user/3721/c37ea1b3-6c92-400a-b3f0-a0782467cdeb.png)

**Transformer模型概述**

在其核心，Transformer模型由编码器和解码器模块堆叠在一起，用于处理输入序列并生成输出序列。以下是架构的高级概述：

**编码器**

+ 编码器模块处理输入序列，提取特征并创建输入的丰富表示。
+ 它由多个编码器层组成，每个层包含自注意力机制和前馈网络。
+ 自注意力机制允许模型同时关注输入序列的不同部分，捕捉依赖关系和关联。
+ 我们将位置编码添加到输入嵌入中，以提供有关序列中标记位置的信息。

**解码器**

+ 解码器模块以编码器的输出作为输入，并生成输出序列。
+ 与编码器类似，它由多个解码器层组成，每个层包含自注意力、编码器-解码器注意力和前馈网络。
+ 除了自注意力外，解码器还包含编码器-解码器注意力，以在生成输出时关注输入序列。
+ 与编码器类似，我们将位置编码添加到输入嵌入中，以提供位置信息。

**连接和标准化**

+ 在编码器和解码器模块的每一层之间，都有残差连接后跟层标准化。
+ 这些机制有助于在网络中流动梯度，并有助于稳定训练。

完整的Transformer模型通过将多个编码器和解码器层堆叠在一起来构建。每个层独立处理输入序列，使模型能够学习分层表示并捕获数据中的复杂模式。编码器将其输出传递给解码器，后者根据输入生成最终的输出序列。

**Transformer模型的实现**

让我们在Python中实现完整的Transformer模型：

```python
# TRANSFORMER的实现
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff,
    max_len, dropout):
        super(Transformer, self).__init__()

        # 定义编码器和解码器的词嵌入层
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 定义位置编码层
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # 定义编码器和解码器的多层堆叠
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout)
        for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout)
        for _ in range(num_layers)])

        # 定义线性层
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    # 生成掩码
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    # 前向传播
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # 编码器输入的词嵌入和位置编码
        encoder_embedding = self.encoder_embedding(src)
        en_positional_encoding = self.positional_encoding(encoder_embedding)
        src_embedded = self.dropout(en_positional_encoding)

        # 解码器输入的词嵌入和位置编码
        decoder_embedding = self.decoder_embedding(tgt)
        de_positional_encoding = self.positional_encoding(decoder_embedding)
        tgt_embedded = self.dropout(de_positional_encoding)

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.linear(dec_output)
        return output

# 示例用法
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_len = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, 
d_ff, max_len, dropout)

# 生成随机示例数据
src_data = torch.randint(1, src_vocab_size, (5, max_len))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (5, max_len))  # (batch_size, seq_length)
transformer(src_data, tgt_data[:, :-1]).shape
```

## 六、模型的训练与评估
训练Transformer模型涉及优化其参数以最小化损失函数，通常使用梯度下降和反向传播。一旦训练完成，就会使用各种指标评估模型的性能，以评估其解决目标任务的有效性。

**训练过程**

梯度下降和反向传播：

+ 在训练期间，将输入序列输入模型，并生成输出序列。
+ 将模型的预测与地面真相进行比较，涉及使用损失函数（例如交叉熵损失）来衡量预测值与实际值之间的差异。
+ 梯度下降用于更新模型的参数，使损失最小化的方向。
+ 优化器根据这些梯度调整参数，迭代更新它们以提高模型性能。

学习率调度：

+ 可以应用学习率调度技术来动态调整训练期间的学习率。
+ 常见策略包括热身计划，其中学习率从低开始逐渐增加，以及衰减计划，其中学习率随时间降低。

**评估指标**

困惑度：

+ 困惑度是用于评估语言模型性能的常见指标，包括Transformer。
+ 它衡量模型对给定标记序列的预测能力。
+ 较低的困惑度值表示更好的性能，理想值接近词汇量大小。

BLEU分数：

+ BLEU（双语评估研究）分数通常用于评估机器翻译文本的质量。
+ 它将生成的翻译与一个或多个由人类翻译人员提供的参考翻译进行比较。
+ BLEU分数范围从0到1，较高的分数表示更好的翻译质量。

## 七、训练和评估的实现
让我们使用PyTorch对Transformer模型进行训练和评估的基本代码实现：

```python
# Transformer 模型的训练和评估
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# 训练循环
transformer.train()

for epoch in range(10):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:]
    .contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"第 {epoch+1} 轮：损失= {loss.item():.4f}")


# 虚拟数据
src_data = torch.randint(1, src_vocab_size, (5, max_len))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (5, max_len))  # (batch_size, seq_length)

# 评估循环
transformer.eval()
with torch.no_grad():
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:]
    .contiguous().view(-1))
    print(f"\n虚拟数据的评估损失= {loss.item():.4f}")
```

## 八、高级主题和应用
Transformers 在自然语言处理（NLP）领域引发了大量先进概念和应用。让我们深入探讨其中一些主题，包括不同的注意力变体、BERT（来自 Transformers 的双向编码器表示）和 GPT（生成式预训练 Transformer），以及它们的实际应用。

**不同的注意力变体**

注意力机制是 Transformer 模型的核心，使其能够专注于输入序列的相关部分。各种注意力变体的提议旨在增强 Transformer 的能力。

1.  缩放点积注意力：是原始 Transformer 模型中使用的标准注意力机制。它将查询和键向量的点积作为注意力分数，同时乘以维度的平方根进行缩放。 
2.  多头注意力：注意力的强大扩展，利用多个注意力头同时捕捉输入序列的不同方面。每个头学习不同的注意力模式，使模型能够并行关注输入的各个部分。 
3.  相对位置编码：引入相对位置编码以更有效地捕捉标记之间的相对位置关系。这种变体增强了模型理解标记之间顺序关系的能力。 

**BERT（来自 Transformers 的双向编码器表示）**

BERT 是一个具有里程碑意义的基于 Transformer 的模型，在 NLP 领域产生了深远影响。它通过掩码语言建模和下一句预测等目标，在大规模文本语料库上进行预训练。BERT 学习了单词的深层上下文表示，捕捉双向上下文，使其在广泛的下游 NLP 任务中表现良好。

代码片段 - BERT 模型:

```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
print(outputs)
```

**GPT（生成式预训练 Transformer）**

GPT 是一个基于 Transformer 的模型，以其生成能力而闻名。与双向的 BERT 不同，GPT 采用仅解码器的架构和自回归训练来生成连贯且上下文相关的文本。研究人员和开发人员已经成功地将 GPT 应用于各种任务，如文本完成、摘要、对话生成等。

代码片段 - GPT 模型:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_text = "Once upon a time, "
inputs=tokenizer(input_text,return_tensors='pt')
output=tokenizer.decode(
    model.generate(
        **inputs,
        max_new_tokens=100,
      )[0],
      skip_special_tokens=True
  )
input_ids = tokenizer(input_text, return_tensors='pt')

print(output)
```

## 九、总结
Transformer 通过其捕捉上下文和理解语言的能力，彻底改变了自然语言处理（NLP）领域。

通过注意力机制、编码器-解码器架构和多头注意力，它们使得诸如机器翻译和情感分析等任务得以在前所未有的规模上实现。随着我们继续探索诸如 BERT 和 GPT 等模型，很明显，Transformer 处于语言理解和生成的前沿。

它们对 NLP 的影响深远，而与 Transformer 一起的发现之旅将揭示出该领域更多令人瞩目的进展。

**研究论文**

+ 《Attention is All You Need》 - Vaswani 等人（2017）
+ 《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》 - Devlin 等人（2018）
+ 《Language Models are Unsupervised Multitask Learners》 - Radford 等人（2019）  
教程：
+ “Attention in transformers, visually explained” - 3Blue1Brown 的 YouTube 教程系列
+ “Transformer Neural Networks, ChatGPT’s foundation” - StatQuest with Josh Starmer 的 YouTube 教程系列

  
完整代码实现的 GitHub

`https://github.com/Ketan3102/Transformers_implementation`

