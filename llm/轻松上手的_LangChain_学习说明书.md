本文为笔者学习 LangChain 时对官方文档以及一系列资料进行一些总结～覆盖对Langchain的核心六大模块的理解与核心使用方法，全文篇幅较长，共计50000+字，可先码住辅助用于学习Langchain。

### **<font style="color:rgb(0, 0, 0);">一、Langchain是什么？</font>**
<font style="color:rgb(62, 71, 83);">如今各类AI模型层出不穷，百花齐放，大佬们开发的速度永远遥遥领先于学习者的学习速度。。为了解放生产力，不让应用层开发人员受限于各语言模型的生产部署中..LangChain横空出世界。</font>

<font style="color:rgb(62, 71, 83);">Langchain可以说是现阶段十分值得学习的一个AI架构，那么究竟它有什么魔法才会配享如此高的地位呢？会不会学习成本很高？不要担心！Langchain虽然功能强大，但其实它就是一个为了提升构建LLM相关应用效率的一个工具，我们也可以将它理解成一个“说明书"，是的，只是一个“说明书”！它</font>**<font style="color:rgb(62, 71, 83);">标准的</font>**<font style="color:rgb(62, 71, 83);">定义了我们在构建一个LLM应用开发时可能会用到的东西。比如说在之前写过的AI文章中介绍的prompt，就可以通过Langchain中的PromptTemplate进行格式化：</font>

```bash
prompt = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
```

<font style="color:rgb(62, 71, 83);">当我们调用ChatPromptTemplate进行标准化时</font>

```python
from langchain.prompts import ChatPromptTemplate
prompt_template=ChatPromptTemplate.from_template(prompt)
print(prompt_template,'ChatPromptTemplate')
```

<font style="color:rgb(62, 71, 83);">该prompt就会被格式化成：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029575102-afedca56-fc1f-4691-bad8-067f8c6ff5dc.png)

<font style="color:rgb(62, 71, 83);">从上述例子，可以直观的看到ChatPromptTemplate可以将prompt中声明的输入变量style和text准确提取出来，使prompt更清晰。当然，Langchain对于prompt的优化不止这一种方式，它还提供了各类其他接口将prompt进一步优化，这里只是举例一个较为基础且直观的方法，让大家感受一下。</font>

<font style="color:rgb(62, 71, 83);">Langchain其实就是在定义多个</font>**<font style="color:rgb(62, 71, 83);">通用类的规范</font>**<font style="color:rgb(62, 71, 83);">，去优化开发AI应用过程中可能用到的各类技术，将它们抽象成多个小元素，当我们构建应用时，直接将这些元素堆积起来，而无需在重复的去研究各"元素"实现的细枝末节。</font>

<font style="color:rgb(62, 71, 83);">  
</font>

### **<font style="color:rgb(0, 0, 0);">二、官方文档Langchain这么长，我怎么看？</font>**
<font style="color:rgb(62, 71, 83);">毋庸置疑，想要学习Langchain最简单直接的方法就是阅读官方文档，先贴一个链接</font>[Langchain官方文档](https://python.langchain.com/docs/get_started/introduction)

<font style="color:rgb(62, 71, 83);">通过文档目录我们可以看到，Langchain由6个module组成，分别是Model IO、Retrieval、Chains、Memory、Agents和Callbacks。</font>

<font style="color:rgb(62, 71, 83);">Model IO：AI应用的核心部分，其中包括输入、Model和输出。</font>

<font style="color:rgb(62, 71, 83);">Retrieval：“检索“——该功能与向量数据密切库相关，是在向量数据库中搜索与问题相关的文档内容。</font>

<font style="color:rgb(62, 71, 83);">Memory：为对话形式的模型存储历史对话记录，在长对话过程中随时将这些历史对话记录重新加载，以保证对话的准确度。</font>

<font style="color:rgb(62, 71, 83);">Chains：虽然通过Model IO、Retrieval和Memory这三大模块可以初步完成应用搭建，但是若想实现一个强大且复杂的应用，还是需要将各模块组合起来，这时就可以利用Chains将其连接起来，从而丰富功能。</font>

<font style="color:rgb(62, 71, 83);">Agents：它可以通过用户的输入，理解用户的意图，返回一个特定的动作类型和参数，从而自主调用相关的工具来满足用户的需求，将应用更加智能化。</font>

<font style="color:rgb(62, 71, 83);">Callbacks: 回调机制可以调用链路追踪，记录日志，帮助开发者更好的调试LLM模型。</font>

<font style="color:rgb(62, 71, 83);">六个module具体的关系如下图所示（图片来源于网络）：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029575136-9bc023fc-d8e4-4aa6-b02f-52218b91d9e6.png)

<font style="color:rgb(62, 71, 83);">好了，说到这我们只要一个一个module去攻破，最后将他们融会贯通，也就成为一名及格的Langchain学习者了。</font>

### **<font style="color:rgb(0, 0, 0);">三、Model IO</font>**
<font style="color:rgb(62, 71, 83);">这一部分可以说是Langchain的核心部分，引用一下之前介绍AI时用过的图，介绍了Model IO内部的一些具体实现原理）</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029575134-3d999280-b40c-46e9-a948-d4f318d5edb3.png)

<font style="color:rgb(62, 71, 83);">由上图可以看出：我们在利用Model IO的时候主要关注的就是</font>**<font style="color:rgb(62, 71, 83);">输入、处理、输出</font>**<font style="color:rgb(62, 71, 83);">这三个步骤。Langchain也是根据这一点去实现Model IO这一模块的，在这一模块中，Langchain针对此模块主要的实现手段为：Prompt(输入)、Language model(处理）、Output Pasers(输出)，Langchain通过一系列的技术手法优化这三步，使得其更加的标准化，我们也无需再关注每一步骤中的具体实现，可以直接通过Langchain提供的API，堆积木式的完善我们应用构建（贴张官方文档的图，可以更清晰的了解）。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029576615-f31d2d53-4ae6-44e6-92b0-5ab83cdf123f.png)

<font style="color:rgb(62, 71, 83);">既然我们无需再关注每一步骤的具体实现，所以使用Langchain的Model IO应用时，主要关注的就是prompt的构建了。下文将主要介绍Langchain中常用的一些prompt构建方法。</font>

#### **<font style="color:rgb(0, 0, 0);">3.1prompt</font>**
<font style="color:rgb(62, 71, 83);">Langchain对于prompt的优化：主要是致力于将其优化成为</font>**<font style="color:rgb(62, 71, 83);">可移植性</font>**<font style="color:rgb(62, 71, 83);">高的Prompt，以便更好的支持各类LLM，无需在切换Model时修改Prompt。 通过官方文档可以看到，Prompt在Langchain被分成了两大类，一类是Prompt template，另一类则是Selectors。</font>

<font style="color:rgb(62, 71, 83);">Propmpt template:这个其实很好理解就是利用Langchain接口将prompt按照template进行一定格式化，针对Prompt进行变量处理以及提示词的组合。</font>

<font style="color:rgb(62, 71, 83);">Selectors: 则是指可以根据不同的条件去选择不同的提示词，或者在不同的情况下通过Selector，选择不同的example去进一步提高Prompt支持能力。</font>

##### **<font style="color:rgb(62, 71, 83);">3.1.1模版格式：</font>**
<font style="color:rgb(62, 71, 83);">在prompt中有两种类型的模版格式，一是f-string，这是十分常见的一类prompt，二是jinja2。</font>

<font style="color:rgb(62, 71, 83);">f-string 是 Python 3.6 以后版本中引入的一种特性，用于在字符串中插入表达式的值。语法简洁，直接利用{}花括号包裹变量或者表达式，即可执行简单的运算，性能较好，但是只限用在py中。</font>

```python
#使用 Python f 字符串模板：
from langchain.prompts import PromptTemplate
fstring_template = """Tell me a {adjective} joke about {content}"""
prompt = PromptTemplate.from_template(fstring_template)
print(prompt.format(adjective="funny", content="chickens"))
# Output: Tell me a funny joke about chickens.
```

<font style="color:rgb(62, 71, 83);">jinja2常被应用于网页开发，与 Flask 和 Django 等框架结合使用。它不仅支持变量替换，还支持其他的控制结构（例如循环和条件语句）以及自定义过滤器和宏等高级功能。此外，它的可用性范围更广，可在多种语境下使用。但与 f-string 不同，使用 jinja2 需要安装相应的库。</font>

```python
#使用 jinja2 模板：
from langchain.prompts import PromptTemplate
jinja2_template = "Tell me a {{ adjective }} joke about {{ content }}"
prompt = PromptTemplate.from_template(jinja2_template, template_format="jinja2")
print(prompt.format(adjective="funny", content="chickens"))
# Output: Tell me a funny joke about chickens.
```

<font style="color:rgb(62, 71, 83);">总结一下:如果只需要基本的字符串插值和格式化，首选f-string ，因为它的语法简洁且无需额外依赖。但如果需要更复杂的模板功能（例如循环、条件、自定义过滤器等），jinja2 更合适。</font>

**<font style="color:rgb(62, 71, 83);">3.1.1.2Propmpt Template：</font>**

<font style="color:rgb(62, 71, 83);">在prompt template这一部分中需要掌握的几个概念：</font>

**<font style="color:rgb(62, 71, 83);">1️⃣</font>****<font style="color:rgb(62, 71, 83);">基本提示模版</font>**<font style="color:rgb(62, 71, 83);">：</font>

<font style="color:rgb(62, 71, 83);">大多是</font>**<font style="color:rgb(62, 71, 83);">字符串</font>**<font style="color:rgb(62, 71, 83);">或者是由</font>**<font style="color:rgb(62, 71, 83);">对话组成的数组对象</font>**<font style="color:rgb(62, 71, 83);">。 对于创建字符串类型的prompt要了解两个概念，一是input_variables 属性，它表示的是prompt所需要输入的变量。二是format，即通过input_variables将prompt格式化。比如利用PromptTemplate进行格式化。</font>

```python
from langchain.prompts import PromptTemplate #用于 PromptTemplate 为字符串提示创建模板。
#默认情况下， PromptTemplate 使用 Python 的 str.format 语法进行模板化;但是可以使用其他模板语法（例如， jinja2 ）
prompt_template = PromptTemplate.from_template("Tell me a {adjective} joke about {content}.")
print(prompt_template.format(adjective="funny", content="chickens"))
```

<font style="color:rgb(62, 71, 83);">Output如下（该例子就是将两个input_variables分别设置为funny和chickens，然后利用format分别进行赋值。若在template中声明了input_variables，利用format进行格式化时就一定要赋值，否则会报错，当在template中未设置input_variables，则会自动忽略。）</font>

```plain
Tell me a funny joke about chickens.
```

<font style="color:rgb(62, 71, 83);">当对</font>**<font style="color:rgb(62, 71, 83);">对话类型</font>**<font style="color:rgb(62, 71, 83);">的prompt进行格式化的时候，可以利用ChatPromptTemplate进行：</font>

```python
#ChatPromptTemplate.from_messages 接受各种消息表示形式。
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])
messages = template.format_messages(
    name="Bob",
    user_input="What is your name?"
)
print(messages)
```

<font style="color:rgb(62, 71, 83);">Output如下（可以看到，ChatPromptTemplate会根据role，对每一句进行标准格式化。除了此类方法，也可以直接指定身份模块如SystemMessage, HumanMessagePromptTemplate进行格式化，这里不再赘述。）</font>

```python
[('system', 'You are a helpful AI bot. Your name is Bob.'),
 ('human', 'Hello, how are you doing?'),
 ('ai', "I'm doing well, thanks!"),
 ('human', 'What is your name?')]
```

**<font style="color:rgb(62, 71, 83);">2️⃣</font>****<font style="color:rgb(62, 71, 83);">部分提示词模版：</font>**

<font style="color:rgb(62, 71, 83);">在生成prompt前就已经提前初始化部分的提示词，实际进一步导入模版的时候只导入除已初始化的变量即可。通常部分提示词模版会被用在全局设置上，如下示例，在正式format前设定foo值为foo，这样在生成最终prompt的时候只需要指定bar的值即可。有两种方法去指定部分提示词：</font>

```python
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(template="{foo}{bar}", input_variables=["foo", "bar"])

# 可以使用 PromptTemplate.partial() 方法创建部分提示模板。
partial_prompt = prompt.partial(foo="foo")
print(partial_prompt.format(bar="baz"))

#也可以只使用分部变量初始化提示。
prompt = PromptTemplate(template="{foo}{bar}", input_variables=["bar"], partial_variables={"foo": "foo"})
print(prompt.format(bar="baz"))
```

<font style="color:rgb(62, 71, 83);">Output如下:</font>

```python
foobaz
foobaz
```

<font style="color:rgb(62, 71, 83);">此外，我们也可以将函数的最终值作为prompt的一部分进行返回，如下例子，如果想在prompt中实时展示当下时间，我们可以直接声明一个函数用来返回当下时间，并最终将该函数拼接到prompt中去：</font>

```python
from datetime import datetime

def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")

prompt = PromptTemplate(
    template="Tell me a {adjective} joke about the day {date}",
    input_variables=["adjective", "date"]
)
partial_prompt = prompt.partial(date=_get_datetime)
print(partial_prompt.format(adjective="funny"))

# 除上述方法，部分函数声明和普通的prompt一样，也可以直接用partial_variables去声明
prompt = PromptTemplate(
    template="Tell me a {adjective} joke about the day {date}",
    input_variables=["adjective"],
    partial_variables={"date": _get_datetime})
```

<font style="color:rgb(62, 71, 83);">Output如下:</font>

```python
Tell me a funny joke about the day 12/08/2022, 16:25:30
```

**<font style="color:rgb(62, 71, 83);">3️⃣</font>****<font style="color:rgb(62, 71, 83);">组成提示词模版：</font>**

<font style="color:rgb(62, 71, 83);">可以通过PromptTemplate.compose()方法将多个提示词组合到一起。如下示例，生成了full_prompt和introduction_prompt进行近一步组合。</font>

```python
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate

full_template = """{introduction}
{example}
"""
full_prompt = PromptTemplate.from_template(full_template)

introduction_template = """You are impersonating Elon Musk."""
introduction_prompt = PromptTemplate.from_template(introduction_template)

example_template = """Here's an example of an interaction """
example_prompt = PromptTemplate.from_template(example_template)

input_prompts = [("introduction", introduction_prompt),
                 ("example", example_prompt),]

pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
```

**<font style="color:rgb(62, 71, 83);">4️⃣</font>****<font style="color:rgb(62, 71, 83);">自定义提示模版：</font>**

<font style="color:rgb(62, 71, 83);">在创建prompt时，我们也可以按照自己的需求去创建自定义的提示模版。官方文档举了一个生成给定名称的函数的英语解释例子，在这个例子中函数名称作为输入，并设置提示格式以提供函数的源代码：</font>

```python
import inspect

# 该函数将返回给定其名称的函数的源代码。 inspect作用就是获取源代码
def get_source_code(function_name):
    # Get the source code of the function
    return inspect.getsource(function_name)

# 测试函数
def test():
    return 1 + 1

from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator

# 初始化字符串prompt
PROMPT = """\
提供一个函数名和源代码并给出函数的相应解释
函数名: {function_name}
源代码:
{source_code}
解释:
"""

class FunctionExplainerPromptTemplate(StringPromptTemplate, BaseModel):
    """一个自定义提示模板，以函数名作为输入，并格式化提示模板以提供函数的源代码。 """
    @validator("input_variables")
    def validate_input_variables(cls, v):
        """验证输入变量是否正确。"""
        if len(v) != 1 or "function_name" not in v:
            raise ValueError("函数名必须是唯一的输入变量。")
        return v

    def format(self, **kwargs) -> str:
        # 获取源代码
        source_code = get_source_code(kwargs["function_name"])
        # 源代码+名字提供给prompt
        prompt = PROMPT.format(
            function_name=kwargs["function_name"].__name__, source_code=source_code)
        return prompt

    def _prompt_type(self):
        return "function-explainer"
```

<font style="color:rgb(62, 71, 83);">FunctionExplainerPromptTemplate接收两个变量一个是prompt，另一个则是传入需要用到的model，该class下面的validate_input_variables用来验证输入量，format函数用来输出格式化后的prompt.</font>

```python
#初始化prompt实例
fn_explainer = FunctionExplainerPromptTemplate(input_variables=["function_name"])

# 定义函数 test_add
def test_add():
    return 1 + 1

# Generate a prompt for the function "test_add"
prompt_1 = fn_explainer.format(function_name=test_add)
print(prompt_1)
```

<font style="color:rgb(62, 71, 83);">Output如下：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029575590-9a40140b-0c90-483d-b028-0d62c8b75fca.png)

**<font style="color:rgb(62, 71, 83);">5️⃣</font>****<font style="color:rgb(62, 71, 83);">少量提示模版：</font>**

<font style="color:rgb(62, 71, 83);">在构建prompt时，可以通过构建一个少量示例列表去进一步格式化prompt，每一个示例表都的结构都为字典，其中键是输入变量，值是输入变量的值。该过程通常先利用PromptTemplate将示例格式化成为字符串，然后创建一个FewShotPromptTemplate对象，用来接收few-shot的示例。官方文档中举例：</font>

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"question": "Who lived longer, Muhammad Ali or Alan Turing?",
     "answer":
     """
    Are follow up questions needed here: Yes.
    Follow up: How old was Muhammad Ali when he died?
    Intermediate answer: Muhammad Ali was 74 years old when he died.
    Follow up: How old was Alan Turing when he died?
    Intermediate answer: Alan Turing was 41 years old when he died.
    So the final answer is: Muhammad Ali
    """},
    {"question": "When was the founder of craigslist born?",
     "answer":
     """
    Are follow up questions needed here: Yes.
    Follow up: Who was the founder of craigslist?
    Intermediate answer: Craigslist was founded by Craig Newmark.
    Follow up: When was Craig Newmark born?
    Intermediate answer: Craig Newmark was born on December 6, 1952.
    So the final answer is: December 6, 1952
    """},
    {"question": "Who was the maternal grandfather of George Washington?",
     "answer":
     """
    Are follow up questions needed here: Yes.
    Follow up: Who was the mother of George Washington?
    Intermediate answer: The mother of George Washington was Mary Ball Washington.
    Follow up: Who was the father of Mary Ball Washington?
    Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
    So the final answer is: Joseph Ball
    """},
    {"question": "Are both the directors of Jaws and Casino Royale from the same country?",
     "answer":
     """
    Are follow up questions needed here: Yes.
    Follow up: Who is the director of Jaws?
    Intermediate Answer: The director of Jaws is Steven Spielberg.
    Follow up: Where is Steven Spielberg from?
    Intermediate Answer: The United States.
    Follow up: Who is the director of Casino Royale?
    Intermediate Answer: The director of Casino Royale is Martin Campbell.
    Follow up: Where is Martin Campbell from?
    Intermediate Answer: New Zealand.
    So the final answer is: No
    """}
]

# 配置一个格式化程序，该格式化程序将prompt格式化为字符串。此格式化程序应该是一个 PromptTemplate 对象。
example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")
print(example_prompt.format(**examples[0]))

# 创建一个选择器来选择最相似的例子
example_selector = SemanticSimilarityExampleSelector(
    examples=examples,
    vector_store=Chroma(),
    embeddings_model=OpenAIEmbeddings(),
    example_prompt=example_prompt
)

# 最后用FewShotPromptTemplate 来创建一个提示词模板，该模板将输入变量作为输入，并将其格式化为包含示例的提示词。
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)
print(prompt)
```

<font style="color:rgb(62, 71, 83);">除了上述普通的字符串模版，聊天模版中也可以采用此类方式构建一个带例子的聊天提示词模版：</font>

```python
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# 这是一个聊天提示词模板，它将输入变量作为输入，并将其格式化为包含示例的提示词。
examples = [{"input": "2+2", "output": "4"}, {"input": "2+3", "output": "5"},]

# 提示词模板，用于格式化每个单独的示例。
example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}"),
     ("ai", "{output}"),])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples)

print(few_shot_prompt.format())
```

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029575617-53b966cf-aff4-4bf5-9e91-046e953df77c.png)

**<font style="color:rgb(62, 71, 83);">6️⃣</font>****<font style="color:rgb(62, 71, 83);">独立化prompt：</font>**

<font style="color:rgb(62, 71, 83);">为了便于共享、存储和加强对prompt的版本控制，可以将想要设定prompt所支持的格式保存为JSON或者YAML格式文件。也可以直接将待格式化的prompt单独存储于一个文件中，通过格式化文件指定相应路径，以更方便用户加载任何类型的提示信息。</font>

<font style="color:rgb(62, 71, 83);">创建json文件：</font>

```python
{
    "_type": "prompt",
    "input_variables": ["adjective", "content"],
    "template": "Tell me a {adjective} joke about {content}."
}
```

<font style="color:rgb(62, 71, 83);">主文件代码：</font>

```plain
from langchain.prompts import load_prompt

prompt = load_prompt("./simple_prompt.json")
print(prompt.format(adjective="funny", content="chickens"))
```

<font style="color:rgb(62, 71, 83);">Output如下：</font>

<font style="color:rgb(62, 71, 83);">Tell me a funny joke about chickens.</font>

<font style="color:rgb(62, 71, 83);">这里是直接在json文件中指定template语句，除此之外也可以将template单独抽离出来，然后在json文件中指定template语句所在的文件路径，以实现更好的区域化，方便管理prompt。</font>

<font style="color:rgb(62, 71, 83);">创建json文件：</font>

```plain
{
  "_type": "prompt",
  "input_variables": ["adjective", "content"],
  "template_path": "./simple_template.txt"
}
```

<font style="color:rgb(62, 71, 83);">simple_template.txt：</font>

```plain
Tell me a {adjective} joke about {content}.
```

<font style="color:rgb(62, 71, 83);">其余部分代码同第一部分介绍，最后的输出结果也是一致的。</font>

**<font style="color:rgb(62, 71, 83);">3.1.1.3Selector：</font>**

<font style="color:rgb(62, 71, 83);">在few shot模块，当我们列举一系列示例值，但不进一步指定返回值，就会返回所有的prompt示例，在实际开发中我们可以使用自定义选择器来选择例子。例如，想要返回一个和新输入的内容最为近似的prompt，这时候就可以去选择与输入最为相似的例子。这里的底层逻辑是利用了SemanticSimilarityExampleSelector这个例子选择器和向量相似度的计算(openAIEmbeddings)以及利用chroma进行数据存储，代码如下：</font>

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 可选的示例列表。
    examples,
    # 用于生成嵌入的嵌入类，这些嵌入用于测量语义相似性。
    OpenAIEmbeddings(),
    # 用于存储嵌入并进行相似性搜索的 VectorStore 类。
    Chroma,
    # 要生成的示例数。
    k=1)
```

<font style="color:rgb(62, 71, 83);">然后我们去输入一条想要构建的prompt，遍历整个示例列表，找到最为合适的example。</font>

```python
# 选择与输入最相似的示例。
question = "Who was the father of Mary Ball Washington?"
selected_examples = example_selector.select_examples({"question": question})
print(f"Examples most similar to the input: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")
```

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029575760-4ae0ad55-05cc-4b66-81e6-ea91256372e4.png)

<font style="color:rgb(62, 71, 83);">此时就可以返回一个最相似的例子。接下来我们可以重新重复few shot的步骤，利用FewShotPromptTemplate去创建一个提示词模版。</font>

<font style="color:rgb(62, 71, 83);">对于聊天类型的few shot的prompt我们也可以采用例子选择器进行格式化：</font>

```python
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "2+4", "output": "6"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
]

# 由于我们使用向量存储来根据语义相似性选择示例，因此我们需要首先填充存储。
to_vectorize = [" ".join(example.values()) for example in examples]

# 这里就单纯理解为将value对应的值提取出来进行格式化即可。

# 创建向量库后，可以创建 example_selector 以表示返回的相似向量的个数
# 注意：您需要先创建一个向量存储库（例如：vectorstore = ...）并填充它，然后将其传递给 SemanticSimilarityExampleSelector。
example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore, k=2)

# 提示词模板将通过将输入传递给 `select_examples` 方法来加载示例
example_selector.select_examples({"input": "horse"})
```

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029575903-eb9bf0cf-d637-44d6-94f0-2aa5b74ea02d.png)

<font style="color:rgb(62, 71, 83);">此时就可以返回两个个最相似的例子。接下来我们可以重复few shot的步骤 利用FewShotChatPromptTemplate去创建一个提示词模版。</font>

<font style="color:rgb(62, 71, 83);">上文中介绍了在利用Langchain进行应用开发时所常用的构建prompt方式，无论哪种方式其最终目的都是为了更方便的去构建prompt，并尽可能的增加其复用性。Langchain提供的prompt相关工具远不止上文这些，在了解了基础能力后可以进一步查阅官方文档找到最适合项目特点的工具，进行prompt格式化。</font>

##### **<font style="color:rgb(0, 0, 0);">3.1.2LLM</font>**
<font style="color:rgb(62, 71, 83);">上除了上文中的prompt，LLM作为langchain中的核心内容，也是我们需要花心思去了解学习的，不过还是那句话，应用层的开发实际上无需到模型底层原理了解的十分透彻，我们更应该关注的是llm的调用形式，Langchain作为一个“工具”它并没有提供自己的LLM，而是提供了一个接口，用于与许多不同类型的LLM进行交互，比如耳熟能详的openai、huggingface或者是cohere等，都可以通过langchain快速调用。</font>

<font style="color:rgb(62, 71, 83);">1.单个调用：直接调用Model对象，传入一串字符串然后直接返回输出值，以openAI为例：</font>

```python
from langchain.llms import OpenAI
llm = OpenAI()
print(llm('你是谁'))
```

<font style="color:rgb(62, 71, 83);">2.批量调用：通过generate可以对字符串列表，进行批量应用Model，使输出更加丰富且完整。</font>

```python
llm_result = llm.generate(["给我背诵一首古诗", "给我讲个100字小故事"]*10)
```

<font style="color:rgb(62, 71, 83);">这时的llm_result会生成一个键为generations的数组，这个数组长度为20项，第一项为古诗、第二项为故事、第三项又为古诗，以此规则排列..</font>

<font style="color:rgb(62, 71, 83);">3.异步接口：asyncio库为LLM提供异步支持，目前支持的LLM为OpenAI、PromptLayerOpenAI、ChatOpenAI 、Anthropic 和 Cohere 受支持。 可以使用agenerate 异步调用 OpenAI LLM。 在代码编写中，如果用了科学上网/魔法，以openAI为例，在异步调用之前，则需要预先将openai的proxy设置成为本地代理（这步很重要，若不设置后续会有报错）</font>

```python
import os
import openai
import asyncio
from langchain.llms import OpenAI

# 设置代理
openai.proxy = os.getenv('https_proxy')

# 定义一个同步方式生成文本的函数
def generate_serially():
    llm = OpenAI(temperature=0.9)  # 创建OpenAI对象，并设置temperature参数为0.9
    for _ in range(10):  # 循环10次
        resp = llm.generate(["Hello, how are you?"])  # 调用generate方法生成文本
        print(resp.generations[0][0].text)  # 打印生成的文本

# 定义一个异步生成文本的函数
async def async_generate(llm):
    resp = await llm.agenerate(["Hello, how are you?"])  # 异步调用agenerate方法生成文本
    print(resp.generations[0][0].text)  # 打印生成的文本

# 定义一个并发（异步）方式生成文本的函数
async def generate_concurrently():
    llm = OpenAI(temperature=0.9)  # 创建OpenAI对象，并设置temperature参数为0.9
    tasks = [async_generate(llm) for _ in range(10)]  # 创建10个异步任务
    await asyncio.gather(*tasks)  # 使用asyncio.gather等待所有异步任务完成
```

<font style="color:rgb(62, 71, 83);">可以用time库去检查运行时间，利用同步调用耗时大概为12s，异步耗时仅有2s。通过这种方式可以大大提速任务执行。</font>

<font style="color:rgb(62, 71, 83);">4.自定义大语言模型：在开发过程中如果遇到需要调用不同的LLM时，可以通过自定义LLM实现效率的提高。自定义LLM时，必须要实现的是_call方法，通过这个方法接受一个字符串、一些可选的索引字，并最终返回一个字符串。除了该方法之外，还可以选择性生成一些方法用于以字典的模式返回该自定义LLM类的各属性。</font>

```python
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from typing import Optional, List, Any, Mapping

class CustomLLM(LLM):  # 这个类 CustomLLM 继承了 LLM 类，并增加了一个新的类变量 n。
    n: int  # 类变量，表示一个整数

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,  # 输入的提示字符串
        stop: Optional[List[str]] = None,  # 可选的停止字符串列表，默认为 None
        run_manager: Optional[CallbackManagerForLLMRun] = None,  # 可选的回调管理器，默认为 None
        **kwargs: Any,
    ) -> str:
    # 如果 stop 参数不为 None，则抛出 ValueError 异常
    if stop is not None:
        raise ValueError("stop kwargs are not permitted.")
return prompt[: self.n]  # 返回 prompt 字符串的前 n 个字符

@property  # 一个属性装饰器，用于获取 _identifying_params 的值
def _identifying_params(self) -> Mapping[str, Any]:
    """Get the identifying parameters."""  # 这个方法的文档字符串，说明这个方法的功能是获取标识参数
    return {"n": self.n}  # 返回一个字典，包含 n 的值
```

<font style="color:rgb(62, 71, 83);">5.测试大语言模型：为了节省我们的成本，当写好一串代码进行测试的时候，通常情况下我们是不希望去真正调用LLM，因为这会消耗token(打工人表示伤不起)，贴心的Langchain则提供给我们一个“假的”大语言模型，以方便我们进行测试。</font>

```python
# 从langchain.llms.fake模块导入FakeListLLM类，此类可能用于模拟或伪造某种行为
from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

# 调用load_tools函数，加载"python_repl"的工具
tools = load_tools(["python_repl"])
# 定义一个响应列表，这些响应可能是模拟LLM的预期响应
responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4"]
# 使用上面定义的responses初始化一个FakeListLLM对象
llm = FakeListLLM(responses=responses)
# 调用initialize_agent函数，使用上面的tools和llm，以及指定的代理类型和verbose参数来初始化一个代理
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
# 调用代理的run方法，传递字符串"whats 2 + 2"作为输入，询问代理2加2的结果
agent.run("whats 2 + 2")
```

<font style="color:rgb(62, 71, 83);">与模拟llm同理，langchain也提供了一个伪类去模拟人类回复，该功能依赖于wikipedia，所以模拟前需要install一下这个库，并且需要设置proxy。这里同fakellm需要依赖agent的三个类，此外它还依赖下面的库：</font>

```python
# 从langchain.llms.human模块导入HumanInputLLM类，此类可能允许人类输入或交互来模拟LLM的行为
from langchain.llms.human import HumanInputLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

# 调用load_tools函数，加载名为"wikipedia"的工具
tools = load_tools(["wikipedia"])

# 初始化一个HumanInputLLM对象，其中prompt_func是一个函数，用于打印提示信息
llm = HumanInputLLM(
    prompt_func=lambda prompt: print(f"\n===PROMPT====\n{prompt}\n=====END OF PROMPT======"))
# 调用initialize_agent函数，使用上面的tools和llm，以及指定的代理类型和verbose参数来初始化一个代理
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# 调用代理的run方法，传递字符串"What is 'Bocchi the Rock!'?"作为输入，询问代理关于'Bocchi the Rock!'的信息
agent.run("What is 'Bocchi the Rock!'?")
```

<font style="color:rgb(62, 71, 83);">6.缓存大语言模型：和测试大语言模型具有一样效果的是缓存大语言模型，通过缓存层可以尽可能的减少API的调用次数，从而节省费用。在Langchain中设置缓存分为两种情况：一是在内存中设置缓存，二是在数据中设置缓存。存储在内存中加载速度较快，但是占用资源并且在关机之后将不再被缓存，在内存中设置缓存示例如下：</font>

```python
from langchain.cache import SQLiteCache
import langchain
from langchain.llms import OpenAI
import time

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

start_time = time.time()  # 记录开始时间
print(llm.predict("用中文讲个笑话"))
end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算总时间
print(f"Predict method took {elapsed_time:.4f} seconds to execute.")
```

<font style="color:rgb(62, 71, 83);">这里的时间大概花费1s+ ，因为被问题放在了内存里，所以在下次调用时几乎不会再耗费时间。</font>

<font style="color:rgb(62, 71, 83);">除了存储在内存中进行缓存，也可以存储在数据库中进行缓存，当开发企业级应用的时候通常都会选择存储在数据库中，不过这种方式的加载速度相较于将缓存存储在内存中更慢一些，不过好处是不占电脑资源，并且存储记录并不会随着关机消失。</font>

```python
from langchain.cache import SQLiteCache
import langchain
from langchain.llms import OpenAI
import time

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

start_time = time.time()  # 记录开始时间
print(llm.predict("用中文讲个笑话"))
end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算总时间
print(f"Predict method took {elapsed_time:.4f} seconds to execute.")
```

<font style="color:rgb(62, 71, 83);">7.跟踪token使用情况（仅限model为openAI）:</font>

```python
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2, cache=None)

with get_openai_callback() as cb:
    result = llm("讲个笑话")
    print(cb)
```

<font style="color:rgb(62, 71, 83);">上述代码直接利用get_openai_callback即可完成对于单条的提问时token的记录，此外对于有多个步骤的链或者agent，langchain也可以追踪到各步骤所耗费的token。</font>

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

llm = OpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

with get_openai_callback() as cb:
    response = agent.run("王菲现在的年龄是多少？")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
```

<font style="color:rgb(62, 71, 83);">8.序列化配置大语言模型：Langchain也提供一种能力用来保存LLM在训练时使用的各类系数，比如template、 model_name等。这类系数通常会被保存在json或者yaml文件中，以json文件为例，配置如下系数，然后利用load_llm方法即可导入：</font>

```python
from langchain.llms.loading import load_llm

llm = load_llm("llm.json")

{
    "model_name": "text-davinci-003",
    "temperature": 0.7,
    "max_tokens": 256,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "n": 1,
    "best_of": 1,
    "request_timeout": None,
    "_type": "openai"
}
```

<font style="color:rgb(62, 71, 83);">亦或者在配置好大模型参数之后，直接利用save方法即可直接保存配置到指定文件中。</font>

<font style="color:rgb(62, 71, 83);">llm.save("llmsave.json")</font>

<font style="color:rgb(62, 71, 83);">9.流式处理大语言模型的响应：流式处理意味着，在接收到第一个数据块后就立即开始处理，而不需要等待整个数据包传输完毕。这种概念应用在LLM中则可达到生成响应时就立刻向用户展示此下的响应，或者在生成响应时处理响应，也就是我们现在看到的和ai对话时逐字输出的效果：可以看到实现还是较为方便的只需要直接调用StreamingStdOutCallbackHandler作为callback即可。</font>

```python
from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
resp = llm("Write me a song about sparkling water.")
```

<font style="color:rgb(62, 71, 83);">可以看到实现还是较为方便的只需要直接调用StreamingStdOutCallbackHandler作为callback即可。</font>

##### **<font style="color:rgb(0, 0, 0);">3.1.3OutputParsers</font>**
<font style="color:rgb(62, 71, 83);">Model返回的内容通常都是字符串的模式，但在实际开发过程中，往往希望model可以返回更直观的内容，Langchain提供的输出解析器则将派上用场。在实现一个输出解析器的过程中，需要实现两种方法：</font><font style="color:rgb(62, 71, 83);">1️⃣</font><font style="color:rgb(62, 71, 83);">获取格式指令：返回一个字符串的方法，其中包含有关如何格式化语言模型输出的说明。</font><font style="color:rgb(62, 71, 83);">2️⃣</font><font style="color:rgb(62, 71, 83);">Parse：一种接收字符串（假设是来自语言模型的响应）并将其解析为某种结构的方法。</font>

<font style="color:rgb(62, 71, 83);">1.列表解析器：利用此解析器可以输出一个用逗号分割的列表。</font>

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

model = OpenAI(temperature=0)

_input = prompt.format(subject="冰淇淋口味")
output = model(_input)

output_parser.parse(output)
```

<font style="color:rgb(62, 71, 83);">2.日期解析器：利用此解析器可以直接将LLM输出解析为日期时间格式。</font>

```python
from langchain.prompts import PromptTemplate
from langchain.output_parsers import DatetimeOutputParser
from langchain.chains import LLMChain
from langchain.llms import OpenAI

output_parser = DatetimeOutputParser()

template = """回答用户的问题:
{question}
{format_instructions}"""

prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

chain = LLMChain(prompt=prompt, llm=OpenAI())

output = chain.run("bitcoin是什么时候成立的？用英文格式输出时间")
```

<font style="color:rgb(62, 71, 83);">3.枚举解析器</font>

```python
from langchain.output_parsers.enum import EnumOutputParser
from enum import Enum

class Colors(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

parser = EnumOutputParser(enum=Colors)
```

<font style="color:rgb(62, 71, 83);">4.自动修复解析器：这类解析器是一种嵌套的形式，如果第一个输出解析器出现错误，就会直接调用另一个一修复错误</font>

```python
# 导入所需的库和模块
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List

# 定义一个表示演员的数据结构，包括他们的名字和他们出演的电影列表
class Actor(BaseModel):
    name: str = Field(description="name of an actor")  # 演员的名字
    film_names: List[str] = Field(description="list of names of films they starred in")  # 他们出演的电影列表

# 定义一个查询，用于提示生成随机演员的电影作品列表
actor_query = "Generate the filmography for a random actor."

# 使用`Actor`模型初始化解析器
parser = PydanticOutputParser(pydantic_object=Actor)

# 定义一个格式错误的字符串数据
misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"

# 使用解析器尝试解析上述数据
try:
    parsed_data = parser.parse(misformatted)
except Exception as e:
    print(f"Error: {e}")
```

<font style="color:rgb(62, 71, 83);">parser.parse(misformatted)</font>

<font style="color:rgb(62, 71, 83);">格式错误的原因是因为json文件需要双引号进行标记，但是这里用了单引号，此时利用该解析器进行解析就会出现报错，但是此时可以利用RetryWithErrorOutputParser进行修复错误，则会正常输出不报错。</font>

```python
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.llms import OpenAI

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=parser, llm=OpenAI(temperature=0))
retry_parser.parse_with_prompt(bad_response, prompt_value)
```

<font style="color:rgb(62, 71, 83);">这里的“Parse_with_prompt”：一种方法，它接受一个字符串（假设是来自语言模型的响应）和一个提示（假设是生成此类响应的提示）并将其解析为某种结构。提示主要在 OutputParser 想要以某种方式重试或修复输出时提供，并且需要来自提示的信息才能执行此操作。  
</font>

### **<font style="color:rgb(0, 0, 0);">四、Retrieval</font>**
<font style="color:rgb(62, 71, 83);">Retrieval直接汉译过来即”检索“。该功能经常被应用于构建一个“私人的知识库”，构建过程更多的是将外部数据存储到知识库中。细化这一模块的主要职能有四部分，其包括数据的获取、整理、存储和查询。如下图：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029575978-e0fa1e6a-c0a7-4172-bef6-e10573baab75.png)

<font style="color:rgb(62, 71, 83);">首先，在该过程中可以从本地/网站/文件等资源库去获取数据，当数据量较小时，我们可以直接进行存储，但当数据量较大的时候，则需要对其进行一定的切片，切分时可以按照数据类型进行切片处理，比如针对文本类数据，可以直接按照字符、段落进行切片；代码类数据则需要进一步细分以保证代码的功能性；此外，除了按照数据类型进行切片处理，也可以直接根据token进行切片。而后利用Vector Stores进行向量存储，其中Embedding完成的就是数据的向量化，虽然这一能力往往被嵌套至大模型中，但是我们也要清楚并不是所有的模型都能直接支持文本向量化这一能力。除此之外的memory、self-hosted以及baas则是指向量存储的三种载体形式，可以选择直接存储于内存中，也可以选择存储上云。最后则利用这些向量化数据进行检索，检索形式可以是直接按照向量相似度去匹配相似内容，也可以直接网络，或者借用其他服务实现检索以及数据的返回。</font>

#### **<font style="color:rgb(0, 0, 0);">4.1向量数据库</font>**
##### **<font style="color:rgb(0, 0, 0);">4.1.1基本概念</font>**
<font style="color:rgb(62, 71, 83);">从上文中我们可以发现，对于retrievers来说，向量数据库发挥着很大的作用，它不仅实现向量的存储也可以通过相似度实现向量的检索，但是向量数据库到底是什么呢？它和普通的数据库有着怎样的区别呢？相信还是有很多同学和我一样有一点点疑惑，所以在介绍langchain在此module方面的能力前，先介绍一下向量数据库，以及它在LLM中所发挥的作用。</font>

<font style="color:rgb(62, 71, 83);">我们在对一个事物进行描述的时候，通常会根据事物的各方面特征进行表述。设想这样一个场景，假设你是一名摄影师，拍了大量的照片。为了方便管理和查找，你决定将这些照片存储到一个数据库中。传统的关系型数据库（如 MySQL、PostgreSQL 等）可以帮助你存储照片的元数据，比如拍摄时间、地点、相机型号等。但是，当你想要根据照片的内容（如颜色、纹理、物体等）进行搜索时，传统数据库可能无法满足你的需求，因为它们通常以数据表的形式存储数据，并使用查询语句进行精确搜索。但向量包含了大量信息，使用查询语句很难精确地找到唯一的向量。</font>

<font style="color:rgb(62, 71, 83);">那么此时，向量数据库就可以派上用场。我们可以构建一个多维的空间使得每张照片特征都存在于这个空间内，并用已有的维度进行表示，比如时间、地点、相机型号、颜色....此照片的信息将作为一个点，存储于其中。以此类推，即可在该空间中构建出无数的点，而后我们将这些点与空间坐标轴的原点相连接，就成为了一条条向量，当这些点变为向量之后，即可利用向量的计算进一步获取更多的信息。当要进行照片的检索时，也会变得更容易更快捷。但在向量数据库中进行检索时，检索并不是唯一的而是查询和目标向量最为相似的</font>**<font style="color:rgb(62, 71, 83);">一些</font>**<font style="color:rgb(62, 71, 83);">向量，具有模糊性。</font>

<font style="color:rgb(62, 71, 83);">那么我们可以延伸思考一下，只要对图片、视频、商品等素材进行向量化，就可以实现以图搜图、视频相关推荐、相似宝贝推荐等功能，那应用在LLM中，小则可直接实现相关问题提示，大则我们完全可以利用此特性去历史对话记录中找到一些最类似的对话，然后重新喂给大模型，这将极大的提高大模型的输出结果的准确性。 为更好的了解向量数据库，接下来将继续介绍向量的几种检索方式，以对向量数据库有一个更深度的了解。</font>

##### **<font style="color:rgb(0, 0, 0);">4.1.2存储方式</font>**
<font style="color:rgb(62, 71, 83);">因为每一个向量所记录的信息量都是比较多的，所以自然而然其所占内存也是很大的，举个例子，如果我们的一个向量维度是256维的，那么该向量所占用的内存大小就是：256*32/8=1024字节，若数据库中共计一千万个向量，则所占内存为10240000000字节，也就是9.54GB，已经是一个很庞大的数目了，而在实际开发中这个规模往往更大，因此解决向量数据库的内存占用问题是重中之重的。我们往往会对每个向量进行压缩，从而缩小其内存占用。常常利用乘积量化方法</font>

<font style="color:rgb(62, 71, 83);">乘积量化：该思想将高维向量分解为多个子向量。例如，将一个D维向量分解为m个子向量，每个子向量的维度为D/m。然后对每个子向量进行量化。对于每个子向量空间，使用聚类算法将子向量分为K个簇，并将簇中心作为量化值。然后，用子向量在簇中的索引来表示原始子向量。这样，每个子向量可以用一个整数（量化索引）来表示。最后将量化索引组合起来表示原始高维向量。对于一个D维向量，可以用m个整数来表示，其中每个整数对应一个子向量的量化索引。此外这类方法不仅可以用于优化存储向量也可以用于优化检索。</font>

##### **<font style="color:rgb(0, 0, 0);">4.1.3检索方式</font>**
<font style="color:rgb(62, 71, 83);">通过上段文字的描述，我们不难发现，向量检索过程可以抽象化为“最近邻问题“，对应的算法就是最近邻搜索算法，具体有如下几种：</font>

<font style="color:rgb(62, 71, 83);">1.暴力搜索：依次比较向量数据库中所有的的向量与目标向量的相似度，然后找出相似度最高一个或一些向量，这样得到的结果质量是极高的，但这对于数据量庞大的数据库来说无疑是十分耗时的。</font>

<font style="color:rgb(62, 71, 83);">2.聚类搜索：这类算法首先初始化K个聚类中心，将数据对象分组成若干个类别或簇（cluster）。其主要目的是根据数据的相似性或距离度量来对数据进行分组，然后根据所选的聚类算法，通过迭代计算来更新聚类结果。例如，在K-means算法中，需要不断更新簇中心并将数据对象分配给最近的簇中心；在DBSCAN算法中，需要根据密度可达性来扩展簇并合并相邻的簇。最后设置一个收敛条件，用于判断聚类过程是否结束。收敛条件可以是迭代次数、簇中心变化幅度等。当满足收敛条件时，聚类过程结束。这样的搜索效率大大提高，但是不可避免会出现遗漏的情况。</font>

<font style="color:rgb(62, 71, 83);">3.位置敏感哈希：此算法首先选择一组位置敏感哈希函数，该函数需要满足一个特性：对于相似的数据点，它们的哈希值发生冲突的概率较高；对于不相似的数据点，它们的哈希值发生冲突的概率较低。而后利用该函数对数据集中的每个数据点进行哈希。将具有相同哈希值的数据点存储在相同的哈希桶中。在检索过程中，对于给定的查询点，首先使用LSH函数计算其哈希值，然后在相应的哈希桶中搜索相似的数据点。最后根据需要，可以在搜索到的候选数据点中进一步计算相似度，以找到最近邻。</font>

<font style="color:rgb(62, 71, 83);">4.分层级的导航小世界算法：这是一种基于图的近似最近邻搜索方法，适用于大规模高维数据集。其核心思想是将数据点组织成一个分层结构的图，使得在高层次上可以快速地找到距离查询点较近的候选点，然后在低层次逐步细化搜索范围，从而加速最近邻搜索过程。</font>

<font style="color:rgb(62, 71, 83);">该算法首先创建一个空的多层图结构。每一层都是一个图，其中节点表示数据点，边表示节点之间的连接关系。最底层包含所有数据点，而上层图只包含部分数据点。每个数据点被分配一个随机的层数，表示该点在哪些层次的图中出现。然后插入数据点：对于每个新插入的数据点，首先确定其层数，然后从最高层开始，将该点插入到相应的图中。插入过程中，需要找到该点在每层的最近邻，并将它们连接起来。同时，还需要更新已有节点的连接关系，以保持图的导航性能。其检索过程是首先在最高层的图中找到一个起始点，然后逐层向下搜索，直到达到底层。在每一层，从当前点出发，沿着边进行搜索，直到找到一个局部最近邻。然后将局部最近邻作为下一层的起始点，继续搜索。最后，在底层找到的结果则为最终结果。</font>

#### **<font style="color:rgb(0, 0, 0);">4.2向量数据库与AI</font>**
<font style="color:rgb(62, 71, 83);">前文中大概介绍了向量数据库是什么以及向量数据库所依赖的一些实现技术，接下来我们来谈论一下向量数据库与大模型之间的关系。为什么说想要用好大模型往往离不开向量数据库呢？对于大模型来讲，处理的数据格式一般都是非结构化数据，如音频、文本、图像..我们以大语言模型为例，在喂一份数据给大模型的时候，数据首先会被转为向量，在上述内容中我们知道如果向量较近那么就表示这两个向量含有的信息更为相似，当大量数据不断被喂到大模型中的时候，语言模型就会逐渐发现词汇间的语义和语法。当用户进行问答的时候，问题输入Model后会基于Transformer架构从每个词出发去找到它与其他词的关系权重，找到权重最重的一组搭配，这一组就为此次问答的答案了。最后再将这组向量返回回来，也就完成了一次问答。当我们把向量数据库接入到AI中，我们就可以通过更新向量数据库的数据，使得大模型能够不断获取并学习到业界最新的知识，而不是将能力局限于预训练的数据中。这种方式要比微调/重新训练大模型的方式节约更多成本。</font>

#### **<font style="color:rgb(0, 0, 0);">4.3DataLoaders</font>**
<font style="color:rgb(62, 71, 83);">为了更好的理解retrieval的功能，在上文中先介绍了一下它所依赖的核心概念——向量数据库，接下来让我们看一下Langhcain中的retrieval是如何发挥作用的。我们已经知道，一般在用户开发（LLM）应用程序，往往会需要使用不在模型训练集中的特定数据去进一步增强大语言模型的能力，这种方法被称为检索增强生成（RAG）。LangChain 提供了一整套工具来实现 RAG 应用程序，首先第一步就是进行文档的相应加载即DocumentLoader：</font>

<font style="color:rgb(62, 71, 83);">LangChain提供了多种文档加载器，支持从各种不同的来源加载文档（例如，私有的存储桶或公共网站），支持的文档类型也十分丰富：如 HTML、PDF 、MarkDown文件等...</font>

<font style="color:rgb(62, 71, 83);">1.加载 md文件：</font>

```python
from langchain.document_loaders import TextLoader
# 创建一个TextLoader实例，指定要加载的Markdown文件路径
loader = TextLoader("./index.md")
# 使用load方法加载文件内容并打印
print(loader.load())
```

<font style="color:rgb(62, 71, 83);">2.加载csv文件：</font>

```python
# 导入CSVLoader类
from langchain.document_loaders.csv_loader import CSVLoader

# 创建CSVLoader实例，指定要加载的CSV文件路径
loader = CSVLoader(file_path='./index.csv')

# 使用load方法加载数据并将其存储在数据变量中
data = loader.load()
```

<font style="color:rgb(62, 71, 83);">3.自定义 csv 解析和加载 指定csv文件的字段名fieldname即可</font>

```python
from langchain.document_loaders.csv_loader import CSVLoader

# 创建CSVLoader实例，指定要加载的CSV文件路径和CSV参数
loader = CSVLoader(file_path='./index.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['title', 'content']
})

# 使用load方法加载数据并将其存储在数据变量中
data = loader.load()
```

<font style="color:rgb(62, 71, 83);">4.可以使用该 source_column 参数指定文件加载的列。</font>

```python
from langchain.document_loaders.csv_loader import CSVLoader

# 创建CSVLoader实例，指定要加载的CSV文件路径和源列名
loader = CSVLoader(file_path='./index.csv', source_column="context")

# 使用load方法加载数据并将其存储在数据变量中
data = loader.load()
```

<font style="color:rgb(62, 71, 83);">除了上述的单个文件加载，我们也可以批量加载一个文件夹内的所有文件，该加载依赖unstructured，所以开始前需要pip一下。如加载md文件就：pip install "unstructured[md]"</font>

```python
# 导入DirectoryLoader类
from langchain.document_loaders import DirectoryLoader

# 创建DirectoryLoader实例，指定要加载的文件夹路径、要加载的文件类型和是否使用多线程
loader = DirectoryLoader('/Users/kyoku/Desktop/LLM/documentstore', glob='**/*.md', use_multithreading=True)

# 使用load方法加载所有文档并将其存储在docs变量中
docs = loader.load()

# 打印加载的文档数量
print(len(docs))

# 导入UnstructuredHTMLLoader类
from langchain.document_loaders import UnstructuredHTMLLoader

# 创建UnstructuredHTMLLoader实例，指定要加载的HTML文件路径
loader = UnstructuredHTMLLoader("./index.html")

# 使用load方法加载HTML文件内容并将其存储在data变量中
data = loader.load()

# 导入BSHTMLLoader类
from langchain.document_loaders import BSHTMLLoader

# 创建BSHTMLLoader实例，指定要加载的HTML文件路径
loader = BSHTMLLoader("./index.html")

# 使用load方法加载HTML文件内容并将其存储在data变量中
data = loader.load()
```

#### **<font style="color:rgb(0, 0, 0);">4.4文本拆分DataTransformers</font>**
<font style="color:rgb(62, 71, 83);">当文件内容成功加载之后，通常会对数据集进行一系列处理，以便更好地适应你的应用。比如说，可能想把长文档分成小块，这样就能更好地放入模型。LangChain 提供了很多现成的文档转换器，可以轻松地拆分、组合、过滤文档，还能进行其他操作。</font>

<font style="color:rgb(62, 71, 83);">虽然上述步骤听起来较为简单，但实际上有很多潜在的复杂性。最好的情况是，把相关的文本片段放在一起。这种“相关性”可能因文本的类型而有所不同。</font>

<font style="color:rgb(62, 71, 83);">Langchain提供了工具RecursiveCharacterTextSplitter用来进行文本的拆分，其运行原理为：首先尝试用第一个字符进行拆分，创建小块。如果有些块太大，它就会尝试下一个字符，以此类推。默认情况下，它会按照 ["\n\n", "\n", " ", ""] 的顺序尝试拆分字符。以下为示例代码：</font>

```python
# 打开一个文本文件并读取内容
with open('./test.txt') as f:
    state_of_the_union = f.read()

# 导入RecursiveCharacterTextSplitter类
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 创建RecursiveCharacterTextSplitter实例，设置块大小、块重叠、长度函数和是否添加开始索引
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True,
)

# 使用create_documents方法创建文档并将其存储在texts变量中
texts = text_splitter.create_documents([state_of_the_union])
```

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029576027-d6469187-a1fa-4826-8f6e-629bc5178dd3.png)

<font style="color:rgb(62, 71, 83);">从输出结果可以看到其是被拆分成了一个数组的形式。</font>

<font style="color:rgb(62, 71, 83);">除了上述的文本拆分，代码拆分也经常被应用于llm应用的构建中：</font>

```python
# 导入所需的类和枚举
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# 定义一个包含Python代码的字符串
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""

# 使用from_language方法创建一个针对Python语言的RecursiveCharacterTextSplitter实例
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)

# 使用create_documents方法创建文档并将其存储在python_docs变量中
python_docs = python_splitter.create_documents([PYTHON_CODE])
```

<font style="color:rgb(62, 71, 83);">调用特定的拆分器可以保证拆分后的代码逻辑，这里我们只要指定不同的Language就可以对不同的语言进行拆分。</font>

#### **<font style="color:rgb(0, 0, 0);">4.5向量检索简单应用</font>**
<font style="color:rgb(62, 71, 83);">在实际开发中我们可以将数据向量化细分为两步：一是将数据向量化(向量化工具：openai的embeding、huggingface的n3d...)，二是将向量化后的数据存储到向量数据库中，常见比较好用的免费向量数据库有Meta的faiss、chrome的chromad以及lance。</font>

<font style="color:rgb(62, 71, 83);">1.高性能：利用 CPU 和 GPU 的并行计算能力，实现了高效的向量索引和查询操作。 2.可扩展性：支持大规模数据集，可以处理数十亿个高维向量的相似性搜索和聚类任务。 3.灵活性：提供了多种索引和搜索算法，可以根据具体需求选择合适的算法。 4.开源：是一个开源项目，可以在 GitHub 上找到其源代码和详细文档。</font>

<font style="color:rgb(62, 71, 83);">安装相关库： pip install faiss-cpu (显卡好的同学也可以install gpu版本)</font>

<font style="color:rgb(62, 71, 83);">准备一个数据集，这个数据集包含一段关于信用卡年费收取和提高信用卡额度的咨询对话。客户向客服提出了关于信用卡年费和额度的问题，客服则详细解答了客户的疑问：</font>

```python
text = """客户：您好，我想咨询一下信用卡的问题。\n客服：您好，欢迎咨询建行信用卡，我是客服小李，请问有什么问题我可以帮您解答吗？\n客户：我想了解一下信用卡的年费如何收取？\n客服：关于信用卡年费的收取，我们会在每年的固定日期为您的信用卡收取年费。当然，如果您在一年内的消费达到一定金额，年费会自动免除。具体的免年费标准，请您查看信用卡合同条款或登录我们的网站查询。\n客户：好的，谢谢。那我还想问一下，如何提高信用卡的额度？\n客服：关于提高信用卡额度，您可以通过以下途径操作：1. 登录建行信用卡官方网站或手机APP，提交在线提额申请；2. 拨打我们的客服热线，按语音提示进行提额申请；3. 您还可以前往附近的建行网点，提交提额申请。在您提交申请后，我们会根据您的信用状况进行审核，审核通过后，您的信用卡额度将会相应提高。\n客户：明白了，非常感谢您的解答。\n客服：您太客气了，很高兴能够帮到您。如果您还有其他问题，请随时联系我们。祝您生活愉快！"""
list_text = text.split('\n')

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

db = FAISS.from_texts(list_text, OpenAIEmbeddings())

query = "信用卡的额度可以提高吗"
docs = db.similarity_search(query)
print(docs[0].page_content)

embedding_vector = OpenAIEmbeddings().embed_query(query)
print(f'embedding_vector：{embedding_vector}')
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)
```

<font style="color:rgb(62, 71, 83);">除了上述直接输出效果最好的结果，也可以按照相似度分数进行输出，不过这里的规则是分数越低，相似度越高。</font>

```python
# 使用带分数的相似性搜索
docs_and_scores = db.similarity_search_with_score(query)

# 打印文档及其相似性分数
for doc, score in docs_and_scores:
    print(f"Document: {doc.page_content}\nScore: {score}\n")
```

<font style="color:rgb(62, 71, 83);">如果每次都要调用embedding无疑太浪费，所以最后我们也可以直接将数据库保存起来，避免重复调用。</font>

```python
# 保存
db.save_local("faiss_index")
# 加载
new_db = FAISS.load_local("faiss_index", OpenAIEmbeddings())
```

<font style="color:rgb(62, 71, 83);">在官网中还介绍了另外两种向量数据库的使用方法，这里不再赘述。</font>

### **<font style="color:rgb(0, 0, 0);">五、Memory</font>**
<font style="color:rgb(62, 71, 83);">Memory——存储历史对话信息。该功能主要会执行两步：1.输入时，从记忆组件中查询相关历史信息，拼接历史信息和用户的输入到提示词中传给LLM。2.自动把LLM返回的内容存储到记忆组件，用于下次查询。</font>

#### **<font style="color:rgb(0, 0, 0);">5.1Memory的基本实现原理：</font>**
<font style="color:rgb(62, 71, 83);">Memory——存储历史对话信息。该功能主要会执行两步：</font>

<font style="color:rgb(62, 71, 83);">1.输入时，从记忆组件中查询相关历史信息，拼接历史信息和用户的输入到提示词中传给LLM。</font>

<font style="color:rgb(62, 71, 83);">2.自动把LLM返回的内容存储到记忆组件，用于下次查询。</font>

<font style="color:rgb(62, 71, 83);">不过，GPT目前就有这个功能了，它已经可以进行多轮对话了，为何我们还要把这个功能拿出来细说呢？在之前介绍prompt的文章中介绍过：在进行多轮对话时，我们会把历史对话内容不断的push到prompt数组中，通俗来讲就是将所有的聊天记录都作为prompt了，以存储的形式实现了大语言模型的“记忆”功能，而大语言模型本身是无状态的，这种方式无疑会较为浪费token，所以开发者不得不将注意力聚焦于</font>**<font style="color:rgb(62, 71, 83);">如何在保证大语言模型功能的基础上尽可能的减少token的使用，</font>**<font style="color:rgb(62, 71, 83);">Memory这个组件也就随之诞生。po一张Memory官网的图：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029576190-5a9b4f1e-8481-4233-a6ea-d0370bc7a79d.png)

<font style="color:rgb(62, 71, 83);">从上图可以看到Memory实现思路还是蛮简单的，就是存储查询，存储的过程我们无需过度思考，无非就是存到内存/数据库，但是读取的过程还是值得我们探讨一番，为什么这么说呢？在上文中已经知道memory的目的其实就是要在保证大语言模型能力的前提下尽可能的减少token消耗，所以我们不能把所有的数据一起丢给大语言模型，这就失去了memory的意义了，不是吗？目前memory常利用以下几种查询策略：</font>

**<font style="color:rgb(62, 71, 83);">1.将会话直接作为prompt喂回给大模型背景，可以称之为buffer。</font>**

**<font style="color:rgb(62, 71, 83);">2.将所有历史消息丢给模型生成一份摘要，再将摘要作为prompt背景，可以称之为summary。</font>**

**<font style="color:rgb(62, 71, 83);">3.利用之前提及的向量数据库，查询相似历史信息，作为prompt背景，可以称之为vector。</font>**

#### **<font style="color:rgb(0, 0, 0);">5.2Memory的使用方式：</font>**
<font style="color:rgb(62, 71, 83);">Memory这一功能的使用方式还是较为简单的，本节将会按照memory的三大分类，依次介绍memory中会被高频使用到的一些工具函数。</font>

##### **<font style="color:rgb(0, 0, 0);">5.2.1Buffer</font>**
**<font style="color:rgb(62, 71, 83);">1️⃣</font>****<font style="color:rgb(62, 71, 83);">ConversationBufferMemory</font>**

<font style="color:rgb(62, 71, 83);">先举例一个最简单的使用方法——直接将内容存储到buffer，无论是单次或是多次存储，其对话内容都会被存储到一个memory：</font>

```python
memory = ConversationBufferMemory() memory.save_context({"input": "你好，我是人类"}, {"output": "你好，我是AI助手"})memory.save_context({"input": "很开心认识你"}, {"output": "我也是"})
```

<font style="color:rgb(62, 71, 83);">存储后可直接输出存储内容：</font>

```python
print(memory.load_memory_variables({}))
# {'history': 'Human: 你好，我是人类\nAI: 你好，我是AI助手\nHuman: 很开心认识你\nAI: 我也是'}
```

##### **<font style="color:rgb(0, 0, 0);">2️⃣</font>****<font style="color:rgb(0, 0, 0);">ConversationBufferWindowMemory</font>**
<font style="color:rgb(62, 71, 83);">ConversationBufferMemory无疑是很简单方便的，但是可以试想一下，当我们与大语言模型进行多次对话时，直接利用buffer存储的话，所占内存量是十分大的，并且消耗的token是十分多的，这时通过ConversationBufferWindowMemory进行窗口缓存的方式就可以解决上述问题。其核心思想：就是保留一个窗口大小的对话，其内容只是</font>**<font style="color:rgb(62, 71, 83);">最近</font>**<font style="color:rgb(62, 71, 83);">的N次对话。在这个工具函数中，可以利用k参数来声明保留的对话记忆，比如k=1时，上述对话内容输出结果就会发生相应的改变：</font>

```python
memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "你好，我是人类"}, {"output": "你好，我是AI助手"})
memory.save_context({"input": "很开心认识你"}, {"output": "我也是"})
```

<font style="color:rgb(62, 71, 83);">只保存了最近的k条记录：</font>

```python
print(memory.load_memory_variables({}))
# {'history': 'Human: 很开心认识你\nAI: 我也是'}
```

<font style="color:rgb(62, 71, 83);">通过内置在Langchain中的缓存窗口(BufferWindow)可以将meomory"记忆"下来。</font>

##### **<font style="color:rgb(0, 0, 0);">3️⃣</font>****<font style="color:rgb(0, 0, 0);">ConversationTokenBufferMemory</font>**
<font style="color:rgb(62, 71, 83);">除了通过设置对话数量控制memory，也可以通过设置token来限制。如果字符数量超出指定数目，它会切掉这个对话的早期部分 以保留与最近的交流相对应的字符数量</font>

```python
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
llm = ChatOpenAI(temperature=0.0)
memory = ConversationTokenBufferMemory(llm=llm,)
memory.save_context({"input": "春眠不觉晓"}, {"output": "处处闻啼鸟"})
memory.save_context({"input": "夜来风雨声"}, {"output": "花落知多少"})
print(memory.load_memory_variables({}))
#{'history': 'AI: 花落知多少。'}
```

##### **<font style="color:rgb(0, 0, 0);">5.2.2Summary</font>**
<font style="color:rgb(62, 71, 83);">对于buffer方式我们不难发现，如果全部保存下来太过浪费，截断时无论是按照对话条数还是token都是无法保证即节省内存或token又保证对话质量的，所以我们可以对其进行summary：</font>

##### **<font style="color:rgb(0, 0, 0);">ConversationSummaryBufferMemory</font>**
<font style="color:rgb(62, 71, 83);">在进行总结时最基础的就是ConversationSummaryBufferMemory这个工具函数，利用该函数时通过设置token从而在清除历史对话时生成一份对话记录：</font>

```python
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=40, return_messages=True)
memory.save_context({"input": "嗨"}, {"output": "你好吗"})
memory.save_context({"input": "没什么特别的，你呢"}, {"output": "我也是"})

messages = memory.chat_memory.messages
previous_summary = ""
print(memory.predict_new_summary(messages, previous_summary))
# 人类和AI都表示没有做什么特别的事
```

<font style="color:rgb(62, 71, 83);">该API通过 predict_new_summary成功的将对话进行了摘要总结。</font>

##### **<font style="color:rgb(62, 71, 83);">5.2.3vector</font>**
<font style="color:rgb(62, 71, 83);">最后来介绍一下vector在memory中的用法，通过VectorStoreRetrieverMemory可以将memory存储到Vector数据库中，每次调用时，就会查找与该记忆关联最高的k个文档，并且不会跟踪交互顺序。不过要注意的是，在利用VectorStoreRetrieverMemory前，我们需要先初始化一个VectorStore，免费向量数据库有Meta的faiss、chrome的chromad以及lance，以faiss为例：</font>

```python
import faiss

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
```

<font style="color:rgb(62, 71, 83);">初始化好一个数据库之后，我们就可以根据该数据库实例化出一个memory：</font>

```python
# 在实际使用中，可以将`k` 设为更高的值，这里使用 k=1 来展示
# 向量查找仍然返回语义相关的信息
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# 当添加到一个代理时，内存对象可以保存来自对话或使用的工具的相关信息
memory.save_context({"input": "我最喜欢的食物是披萨"}, {"output": "好的，我知道了"})
memory.save_context({"input": "我最喜欢的运动是足球"}, {"output": "..."})
memory.save_context({"input": "我不喜欢凯尔特人队"}, {"output": "好的"}) 
print(memory.load_memory_variables({"prompt": "我应该看什么运动?"})["history"])
```

<font style="color:rgb(62, 71, 83);">这时便会根据向量数据库检索后输出memory结果</font>

```python
{
    'history': [
        {
            'input': '我最喜欢的运动是足球',
            'output': '...'
        }
    ]
}
```

<font style="color:rgb(62, 71, 83);">这表示在与用户的对话历史中，语义上与 "我应该看什么运动?" 最相关的是 "我最喜欢的运动是足球" 这个对话。更复杂一点可以通过conversationchain进行多轮对话：</font>

```python
llm = OpenAI(temperature=0) # 可以是任何有效的LLM
_DEFAULT_TEMPLATE = """以下是一个人类与AI之间的友好对话。AI非常健谈，并从其上下文中提供大量具体细节。如果AI不知道问题的答案，它会诚实地说不知道。

之前对话的相关部分：
{history}

（如果不相关，您不需要使用这些信息）

当前对话：
人类：{input}
AI："""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)
conversation_with_summary = ConversationChain(
    llm=llm, 
    prompt=PROMPT,
    # 我们为测试目的设置了一个非常低的max_token_limit。
    memory=memory,
    verbose=True
)
conversation_with_summary.predict(input="嗨，我叫Perry，你好吗？")
# 输出："> Entering new ConversationChain chain...
# Prompt after formatting:
# ...
# > Finished chain.
# " 嗨，Perry，我很好。你呢？"

# 这里，与篮球相关的内容被提及
conversation_with_summary.predict(input="我最喜欢的运动是什么？")
# 输出："> Entering new ConversationChain chain...
# ...
# > Finished chain.
# ' 你之前告诉我你最喜欢的运动是足球。'"

# 尽管语言模型是无状态的，但由于获取到了相关的记忆，它可以“推理”出时间。
# 为记忆和数据加上时间戳通常是有用的，以便让代理确定时间相关性
conversation_with_summary.predict(input="我的最喜欢的食物是什么？")
# 输出："> Entering new ConversationChain chain...
# ...
# > Finished chain.
# ' 你说你最喜欢的食物是披萨。'"

# 对话中的记忆被自动存储，
# 由于这个查询与上面的介绍聊天最匹配，
# 代理能够“记住”用户的名字。
conversation_with_summary.predict(input="我的名字是什么？")
# 输出："> Entering new ConversationChain chain...
# ...
# > Finished chain.
# ' 你的名字是Perry。'"
```

<font style="color:rgb(62, 71, 83);">conversation_with_summary这个实例使用了一个内存对象（memory）来存储与用户的对话历史。这使得AI可以在后续的对话中引用先前的上下文，从而提供更准确和相关的回答。</font>

<font style="color:rgb(62, 71, 83);">在Langchain中memory属于较为简单的一模块，小型开发中常常使用summary类型，对于大一点的开发来说，最常见的就是利用向量数据库进行数据的存储，并在ai模型给出输出时到该数据库中检索出相似性最高的内容。</font>

### **<font style="color:rgb(0, 0, 0);">六、Chains</font>**
<font style="color:rgb(62, 71, 83);">如果把用Langchain构建AI应用的过程比作“积木模型”的搭建与拼接，那么Chain可以说是该模型搭建过程中的骨骼部分，通过它将各模块快速组合在一起就可以快速搭建一个应用。Chain的使用方式也是通过接口的直接调用，在本文中将Chain分为三种类型，从简单到复杂依次介绍按照首先以一个简单的示例，来直观的感受Chain的作用：</font>

#### **<font style="color:rgb(0, 0, 0);">6.1 LLMChains:</font>**
<font style="color:rgb(62, 71, 83);">这种类型的Chain应用起来很简单也可以说是后续要介绍的Chain的基础，但其功能是足够强大的。通过LLMChain可以直接将数据、prompt、以及想要应用的Model串到一起，以一个简单的例子来感受LLMChain。</font>

```python
from langchain import PromptTemplate, OpenAI, LLMChain

prompt_template = "What is a good name for a company that makes {product}?"

llm = OpenAI(temperature=0)
chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)
print(chain("colorful socks")) 
# 输出结果'Socktastic!'
```

<font style="color:rgb(62, 71, 83);">在这个示例中，我们首先初始化了一个prompt的字符串模版，并初始化大语言模型，然后利用Chain将模型运行起来。在「Chain将模型运行起来」这个过程中：Chain将会格式化提示词，然后将它传递给LLM。回忆一下，在之前的</font>[ai入门篇](https://km.woa.com/articles/show/586609)<font style="color:rgb(62, 71, 83);">中，对于每个model的使用，我们需要针对这个model去进行一系列初始化、实例化等操作。而用了chain之后，我们无需再关注model本身。</font>

#### **<font style="color:rgb(0, 0, 0);">6.2 Sequential Chains:</font>**
<font style="color:rgb(62, 71, 83);">不同于基本的LLMChain，Sequential chain（序列链）是由一系列的链组合而成的，序列链有两种类型，一种是单个输入输出/另一个则是多个输入输出。先来看第一种单个输入输出的示例代码：</font>

##### **<font style="color:rgb(0, 0, 0);">1.单个输入输出</font>**
<font style="color:rgb(62, 71, 83);">在这个示例中，创建了两条chain，并且让第一条chain接收一个虚构剧本的标题，输出该剧本的概要，作为第二条chain的输入，然后生成一个虚构评论。通过sequential chains可以简单的实现这一需求。</font>

<font style="color:rgb(62, 71, 83);">第一条chain：</font>

```python
# This is an LLMChain to write a synopsis given a title of a play.
from langchain import PromptTemplate, OpenAI, LLMChain

llm = OpenAI(temperature=.7)
template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)
```

<font style="color:rgb(62, 71, 83);">第二条chain：</font>

```python
# This is an LLMChain to write a review of a play given a synopsis.
from langchain import PromptTemplate, OpenAI, LLMChain

llm = OpenAI(temperature=.7)
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)
```

<font style="color:rgb(62, 71, 83);">最后利用SimpleSequentialChain即可将两个chain直接串联起来：</font>

```python
from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)
print(review = overall_chain.run("Tragedy at sunset on the beach"))
```

<font style="color:rgb(62, 71, 83);">可以看到对于单个输入输出的顺序链，就是将两个chain作为参数传给simplesequentialchain即可，无需复杂的声明。</font>

##### **<font style="color:rgb(0, 0, 0);">2.多个输入输出</font>**
<font style="color:rgb(62, 71, 83);">除了单个输入输出的模式，顺序链还支持更为复杂的多个输入输出，对于多输入输出模式来说，最应该需要关注的就是输入关键字和输出关键字，它们需要十分的精准，才能够保证chain的识别与应用，依旧以一个demo为例：</font>

```python
from langchain import PromptTemplate, OpenAI, LLMChain

llm = OpenAI(temperature=.7)
template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

Title: {title}
Era: {era}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title", 'era'], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")
#第一条chain
```

```python
from langchain import PromptTemplate, OpenAI, LLMChain

llm = OpenAI(temperature=.7)
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")
#第二条chain
```

```python
from langchain.chains import SequentialChain

overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["era", "title"],
    # Here we return multiple variables
    output_variables=["synopsis", "review"],
    verbose=True)
#第三条chain
```

```plain
overall_chain({"title": "Tragedy at sunset on the beach", "era": "Victorian England"})
```

<font style="color:rgb(62, 71, 83);">对于每一个chain在定义的时候，都需要关注其output_key 、和input_variables，按照顺序将其指定清楚。最终在运行chain时我们只需要指定第一个chain中需要声明的变量。</font>

##### **<font style="color:rgb(0, 0, 0);">6.3RouterChains:</font>**
<font style="color:rgb(62, 71, 83);">最后介绍一个经常会用到的场景，比如我们目前有三类chain，分别对应三种学科的问题解答。我们的输入内容也是与这三种学科对应，但是随机的，比如第一次输入数学问题、第二次有可能是历史问题... 这时候期待的效果是：可以根据输入的内容是什么，自动将其应用到对应的子链中。Router Chain就为我们提供了这样一种能力，它会首先决定将要传递下去的子链，然后把输入传递给那个链。并且在设置的时候需要注意为其设置默认chain，以兼容输入内容不满足任意一项时的情况。</font>

```python
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""
```

<font style="color:rgb(62, 71, 83);">如上有一个物理学和数学的prompt：</font>

```python
prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template
    }
]
```

<font style="color:rgb(62, 71, 83);">然后，需要声明这两个prompt的基本信息。</font>

```python
from langchain import ConversationChain, LLMChain, PromptTemplate, OpenAI
llm = OpenAI()
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

default_chain = ConversationChain(llm=llm, output_key="text")
```

<font style="color:rgb(62, 71, 83);">最后将其运行到routerchain中即可，我们此时在输入的时候chain就会根据input的内容进行相应的选择最为合适的prompt。</font>

```python
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

# Create a list of destinations
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# Create a router template
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)


router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)
print(chain.run('什么是黑体辐射'))
```

<font style="color:rgb(62, 71, 83);"></font>

### **<font style="color:rgb(0, 0, 0);">七、Agents</font>**
<font style="color:rgb(62, 71, 83);">Agents这一模块在langchain的使用过程中也是十分重要的，官方文档是这样定义它的“The core idea of agents is to use a language model to choose a sequence of actions to take. In chains, a sequence of actions is hardcoded (in code). In agents, a language model is used as a reasoning engine to determine which actions to take and in which order.”也就是说，在使用Agents时，其行为以及行为的顺序是由LLM的推理机制决定的，并不是像传统的程序一样，由核心代码预定义好去运行的。</font>

<font style="color:rgb(62, 71, 83);">举一个例子来对比一下，对于传统的程序，我们可以想象这样一个场景：一个王子需要经历3个关卡，才可以救到公主，那么王子就必须按部就班的走一条确定的路线，一步步去完成这三关，才可以救到公主，他不可以跳过或者修改关卡本身。但对于Agents来说，我们可以将其想象成一个刚出生的原始人类，随着大脑的日渐成熟和身体的不断发育，该人类将会逐步拥有决策能力和记忆能力，这时想象该人类处于一种饥饿状态，那么他就需要吃饭。此时，他刚好走到小河边，通过“记忆”模块，认知到河里的“鱼”是可以作为食物的，那么他此时就会巧妙的利用自己身边的工具——鱼钩，进行钓鱼，然后再利用火，将鱼烤熟。第二天，他又饿了，这时他在丛林里散步，遇到了一头野猪，通过“记忆”模块，认知到“野猪”也是可以作为食物的，由于野猪的体型较大，于是他选取了更具杀伤力的长矛进行狩猎。从他这两次狩猎的经历，我们可以发现，他并不是按照预先设定好的流程，使用固定的工具去捕固定的猎物，而是根据环境的变化选择合适的猎物，又根据猎物的种类，去决策使用的狩猎工具。这一过程完美的利用了自己的决策、记忆系统，并辅助利用工具，从而做出一系列反应去解决问题。以一个数学公式来表示，可以说Agents=LLM（决策）+Memory（记忆）+Tools（执行）。</font>

<font style="color:rgb(62, 71, 83);">通过上述的例子，相信你已经清楚的认知到Agents与传统程序比起来，其更加灵活，通过不同的搭配，往往会达到令人意想不到的效果，现在就用代码来实操感受一下Agents的实际应用方式，下文的示例代码主要实现的功能是——给予Agent一个题目，让Agent生成一篇论文。</font>

<font style="color:rgb(62, 71, 83);">在该示例中，我们肯定是要示例化Agents，示例化一个Agents时需要关注上文中所描述的它的三要素：LLM、Memory和tools，其代码如下：</font>

```plain
# 初始化 agent
agent = initialize_agent(
    tools,  # 配置工具集
    llm,  # 配置大语言模型 负责决策
    agent=AgentType.OPENAI_FUNCTIONS,  # 设置 agent 类型 
    agent_kwargs=agent_kwargs,  # 设定 agent 角色
    verbose=True,
    memory=memory, # 配置记忆模式 )
```

#### **<font style="color:rgb(0, 0, 0);">7.1tools相关的配置介绍</font>**
<font style="color:rgb(62, 71, 83);">首先是配置工具集tools，如下列代码，可以看到这是一个二元数组，也就意味着本示例中的Agents依赖两个工具。</font>

```python
from langchain.agents import initialize_agent, Tool
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]
```

<font style="color:rgb(62, 71, 83);">先看第一个工具：在配置工具时，需要声明工具依赖的函数，由于该示例实现的功能为依赖网络收集相应的信息，然后汇总成一篇论文，所以创建了一个search函数，这个函数用于调用Google搜索。它接受一个查询参数，然后将查询发送给Serper API。API的响应会被打印出来并返回。</font>

```python
# 调用 Google search by Serper
def search(query):
    serper_google_url = os.getenv("SERPER_GOOGLE_URL")

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", serper_google_url, headers=headers, data=payload)

    print(f'Google 搜索结果: \n {response.text}')
    return response.text
```

<font style="color:rgb(62, 71, 83);">再来看一下所依赖的第二个工具函数，这里用了另一种声明工具的方式Class声明—— ScrapeWebsiteTool()，它有以下几个属性和方法：</font>

```python
class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, target: str, url: str):
        return scrape_website(target, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")
```

<font style="color:rgb(62, 71, 83);">1.name：工具的名称，这里是 "scrape_website"。 2.description：工具的描述。 args_schema：工具的参数模式，这里是 ScrapeWebsiteInput 类，表示这个工具需要的输入参数，声明代码如下，这是一个基于Pydantic的模型类，用于定义 scrape_website 函数的输入参数。它有两个字段：target 和 url，分别表示用户给agent的目标和任务以及需要被爬取的网站的URL。</font>

```python
class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    target: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")
```

<font style="color:rgb(62, 71, 83);">_run 方法：这是工具的主要执行函数，它接收一个目标和一个URL作为参数，然后调用 scrape_website 函数来爬取网站并返回结果。scrape_website 函数根据给定的目标和URL爬取网页内容。首先，它发送一个HTTP请求来获取网页的内容。如果请求成功，它会使用BeautifulSoup库来解析HTML内容并提取文本。如果文本长度超过5000个字符，它会调用 summary 函数来对内容进行摘要。否则，它将直接返回提取到的文本。其代码如下：</font>

```python
# 根据 url 爬取网页内容，给出最终解答
# target ：分配给 agent 的初始任务
# url ： Agent 在完成以上目标时所需要的URL，完全由Agent自主决定并且选取，其内容或是中间步骤需要，或是最终解答需要
def scrape_website(target: str, url: str):
    print(f"开始爬取： {url}...")

    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    payload = json.dumps({
        "url": url
    })

    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=payload)

    # 如果返回成功
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("爬取的具体内容:", text)

        # 控制返回内容长度，如果内容太长就需要切片分别总结处理
        if len(text) > 5000:
            # 总结爬取的返回内容
            output = summary(target, text)
            return output
        else:
            return text
    else:
        print(f"HTTP请求错误，错误码为{response.status_code}")
```

<font style="color:rgb(62, 71, 83);">从上述代码中我们可以看到其还依赖一个summary 函数，用此函数解决内容过长的问题，这个函数使用Map-Reduce方法对长文本进行摘要。它首先初始化了一个大语言模型（llm），然后定义了一个大文本切割器（text_splitter）。接下来，它创建了一个摘要链（summary_chain），并使用这个链对输入文档进行摘要。</font>

```python
# 如果需要处理的内容过长，先切片分别处理，再综合总结
# 使用 Map-Reduce 方式
def summary(target, content):
    # model list ： https://platform.openai.com/docs/models
    # gpt-4-32k   gpt-3.5-turbo-16k-0613
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    # 定义大文本切割器
    # chunk_overlap 是一个在使用 OpenAI 的 GPT-3 或 GPT-4 API 时可能会遇到的参数，特别是需要处理长文本时。
    # 该参数用于控制文本块（chunks）之间的重叠量。
    # 上下文维护：重叠确保模型在处理后续块时有足够的上下文信息。
    # 连贯性：它有助于生成更连贯和一致的输出，因为模型可以“记住”前一个块的部分内容。
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=200)

    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {target}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "target"])

    summary_chain
```

<font style="color:rgb(62, 71, 83);">_arun 方法：这是一个异步版本的 _run 方法，这里没有实现，如果调用会抛出一个 NotImplementedError 异常。</font>

#### **<font style="color:rgb(0, 0, 0);">7.2LLM的配置介绍</font>**
```python
# 初始化大语言模型，负责决策
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
```

<font style="color:rgb(62, 71, 83);">这段代码初始化了一个名为 llm 的大语言模型对象，它是 ChatOpenAI 类的实例。ChatOpenAI 类用于与大语言模型（如GPT-3）进行交互，以生成决策和回答。在初始化 ChatOpenAI 对象时，提供了以下参数：</font>

<font style="color:rgb(62, 71, 83);">1.temperature：一个浮点数，表示生成文本时的温度。温度值越高，生成的文本将越随机和多样；温度值越低，生成的文本将越确定和一致。在这里设置为 0，因为本demo的目的为生成一个论文，所以我们并不希望大模型有较多的可变性，而是希望生成非常确定和一致的回答。 2.model：一个字符串，表示要使用的大语言模型的名称。在这里，我们设置为 "gpt-3.5-turbo-16k-0613"，表示使用 GPT-3.5 Turbo 模型。</font>

#### **<font style="color:rgb(0, 0, 0);">7.3Agent类型及角色相关的配置介绍</font>**
<font style="color:rgb(62, 71, 83);">首先来看一下AgentType这个变量的初始化，这里是用来设置agent类型的一个参数，具体可以参考官网：</font>[AgentType](https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent)

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029576331-dd91c198-d4d9-4e6d-bff9-4e93b9d71ed9.png)

<font style="color:rgb(62, 71, 83);">可以看到官网里列举了7中agent类型，可以根据自己的需求进行选择，在本示例中选用的是第一种类型OpenAi functions。此外，还要设定agent角色以及记忆模式：</font>

```python
# 初始化agents的详细描述
system_message = SystemMessage(
    content="""您是一位世界级的研究员，可以对任何主题进行详细研究并产生基于事实的结果；
            您不会凭空捏造事实，您会尽最大努力收集事实和数据来支持研究。

            请确保按照以下规则完成上述目标：
            1/ 您应该进行足够的研究，尽可能收集关于目标的尽可能多的信息
            2/ 如果有相关链接和文章的网址，您将抓取它以收集更多信息
            3/ 在抓取和搜索之后，您应该思考“根据我收集到的数据，是否有新的东西需要我搜索和抓取以提高研究质量？”如果答案是肯定的，继续；但不要进行超过5次迭代
            4/ 您不应该捏造事实，您只应该编写您收集到的事实和数据
            5/ 在最终输出中，您应该包括所有参考数据和链接以支持您的研究；您应该包括所有参考数据和链接以支持您的研究
            6/ 在最终输出中，您应该包括所有参考数据和链接以支持您的研究；您应该包括所有参考数据和链接以支持您的研究"""
)
# 初始化 agent 角色模板
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

# 初始化记忆类型
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=300)
```

<font style="color:rgb(62, 71, 83);">1️⃣</font><font style="color:rgb(62, 71, 83);">在设置agent_kwargs时："extra_prompt_messages"：这个键对应的值是一个包含 MessagesPlaceholder 对象的列表。这个对象的 variable_name 属性设置为 "memory"，表示我们希望在构建 agent 的提示时，将 memory 变量的内容插入到提示中。"system_message"：这个键对应的值是一个 SystemMessage 对象，它包含了 agent 的角色描述和任务要求。</font>

#### **<font style="color:rgb(0, 0, 0);">7.4Memory的配置介绍</font>**
```python
# 初始化记忆类型
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=300)
```

<font style="color:rgb(62, 71, 83);">在设置 memory 的记忆类型对象时：利用了 ConversationSummaryBufferMemory 类的实例。该类用于在与AI助手的对话中缓存和管理信息。在初始化这个对象时，提供了以下参数：1.memory_key：一个字符串，表示这个记忆对象的键。在这里设置为 "memory"。2.return_messages：一个布尔值，表示是否在返回的消息中包含记忆内容。在这里设置为 True，表示希望在返回的消息中包含记忆内容。3.llm：对应的大语言模型对象，这里是之前初始化的 llm 对象。这个参数用于指定在处理记忆内容时使用的大语言模型。4。max_token_limit：一个整数，表示记忆缓存的最大令牌限制。在这里设置为 300，表示希望缓存的记忆内容最多包含 300 个token。</font>

#### **<font style="color:rgb(0, 0, 0);">7.5依赖的环境包倒入以及启动主函数</font>**
<font style="color:rgb(62, 71, 83);">这里导入所需库：这段代码导入了一系列所需的库，包括os、dotenv、langchain相关库、requests、BeautifulSoup、json和streamlit。</font>

```python
import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage

from typing import Type
from bs4 import BeautifulSoup
import requests
import json

import streamlit as st

# 加载必要的参数
load_dotenv()
serper_api_key=os.getenv("SERPER_API_KEY")
browserless_api_key=os.getenv("BROWSERLESS_API_KEY")
openai_api_key=os.getenv("OPENAI_API_KEY")
```

<font style="color:rgb(62, 71, 83);">main 函数：这是streamlit应用的主函数。它首先设置了页面的标题和图标，然后创建了一些header，并提供一个文本输入框让用户输入查询。当用户输入查询后，它会调用agent来处理这个查询，并将结果显示在页面上。</font>

```python
def main():
    st.set_page_config(page_title="AI Assistant Agent", page_icon=":dolphin:")

    st.header("LangChain 实例讲解 3 -- Agent", divider='rainbow')
    st.header("AI Agent :blue[助理] :dolphin:")

    query = st.text_input("请提问题和需求：")

    if query:
        st.write(f"开始收集和总结资料 【 {query}】 请稍等")

        result = agent({"input": query})

        st.info(result['output'])
```

<font style="color:rgb(62, 71, 83);">至此Agent的使用示例代码就描述完毕了，我们可以看到，其实Agents的功能就是其会自主的去选择并利用最合适的工具，从而解决问题，我们提供的Tools越丰富，则其功能越强大。</font>

### **<font style="color:rgb(62, 71, 83);">八、Callbacks</font>**
<font style="color:rgb(62, 71, 83);">Callbacks对于程序员们应该都不陌生，就是一个回调函数，这个函数允许我们在LLM的各个阶段使用各种各样的“钩子”，从而达实现日志的记录、监控以及流式传输等功能。在Langchain中，该回掉函数是通过继承 BaseCallbackHandler 来实现的，该接口对于每一个订阅事件都声明了一个回掉函数。它的子类也就可以通过继承它实现事件的处理。如官网所示：</font>

```python
class BaseCallbackHandler:
    """Base callback handler that can be used to handle callbacks from langchain."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
    """Run when LLM starts running."""

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
    """Run when Chat Model starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
    """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
    """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
    """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
    """Run when tool starts running."""

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
    """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
```

<font style="color:rgb(62, 71, 83);">这个类包含了一系列方法，这些方法在 langchain 的不同阶段被调用，以便在处理过程中执行自定义操作。参考源码</font>[BaseCallbackHandler](https://github.com/langchain-ai/langchain/blob/v0.0.235/langchain/callbacks/base.py#L225)<font style="color:rgb(62, 71, 83);">：</font>

<font style="color:rgb(62, 71, 83);">on_llm_start: 当大语言模型（LLM）开始运行时调用。 on_chat_model_start: 当聊天模型开始运行时调用。 on_llm_new_token: 当有新的LLM令牌时调用。仅在启用流式处理时可用。 on_llm_end: 当LLM运行结束时调用。 on_llm_error: 当LLM出现错误时调用。 on_chain_start: 当链开始运行时调用。 on_chain_end: 当链运行结束时调用。 on_chain_error: 当链出现错误时调用。 on_tool_start: 当工具开始运行时调用。 on_tool_end: 当工具运行结束时调用。 on_tool_error: 当工具出现错误时调用。 on_text: 当处理任意文本时调用。 on_agent_action: 当代理执行操作时调用。 on_agent_finish: 当代理结束时调用。</font>

#### **<font style="color:rgb(0, 0, 0);">8.1基础使用方式StdOutCallbackHandler</font>**
<font style="color:rgb(62, 71, 83);">StdOutCallbackHandler 是 LangChain 支持的最基本的处理器，它继承自 BaseCallbackHandler。这个处理器将所有回调信息打印到标准输出，对于调试非常有用。以下是如何使用 StdOutCallbackHandler 的示例：</font>

```python
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

handler = StdOutCallbackHandler()
llm = OpenAI()
prompt = PromptTemplate.from_template("Who is {name}?")
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
chain.run(name="Super Mario")
```

<font style="color:rgb(62, 71, 83);">在这个示例中，我们首先从 langchain.callbacks 模块导入了 StdOutCallbackHandler 类。然后，创建了一个 StdOutCallbackHandler 实例，并将其赋值给变量 handler。接下来，导入了 LLMChain、OpenAI 和 PromptTemplate 类，并创建了相应的实例。在创建 LLMChain 实例时，将 callbacks 参数设置为一个包含 handler 的列表。这样，当链运行时，所有的回调信息都会被打印到标准输出。最后，使用 chain.run() 方法运行链，并传入参数 name="Super Mario"。在链运行过程中，所有的回调信息将被 StdOutCallbackHandler 处理并打印到标准输出。</font>

#### **<font style="color:rgb(0, 0, 0);">8.2自定义回调处理器</font>**
```python
from langchain.callbacks.base import BaseCallbackHandler
import time

class TimerHandler(BaseCallbackHandler):

    def __init__(self) -> None:
        super().__init__()
        self.previous_ms = None
        self.durations = []

    def current_ms(self):
        return int(time.time() * 1000 + time.perf_counter() % 1 * 1000)

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        self.previous_ms = self.current_ms()

    def on_chain_end(self, outputs, **kwargs) -> None:
        if self.previous_ms:
            duration = self.current_ms() - self.previous_ms
            self.durations.append(duration)

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self.previous_ms = self.current_ms()

    def on_llm_end(self, response, **kwargs) -> None:
        if self.previous_ms:
            duration = self.current_ms() - self.previous_ms
            self.durations.append(duration)

llm = OpenAI()
timerHandler = TimerHandler()
prompt = PromptTemplate.from_template("What is the HEX code of color {color_name}?")
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[timerHandler])
response = chain.run(color_name="blue")
print(response)
response = chain.run(color_name="purple")
print(response)
```

<font style="color:rgb(62, 71, 83);">这个示例展示了如何通过继承 BaseCallbackHandler 来实现自定义的回调处理器。在这个例子中，创建了一个名为 TimerHandler 的自定义处理器，它用于跟踪 Chain 或 LLM 交互的起止时间，并统计每次交互的处理耗时。从 langchain.callbacks.base 模块导入 BaseCallbackHandler 类。导入 time 模块，用于处理时间相关操作。</font>

<font style="color:rgb(62, 71, 83);">定义 TimerHandler 类，继承自 BaseCallbackHandler。在 TimerHandler 类的</font><font style="color:rgb(62, 71, 83);"> </font>**<font style="color:rgb(62, 71, 83);">init</font>**<font style="color:rgb(62, 71, 83);"> </font><font style="color:rgb(62, 71, 83);">方法中，初始化 previous_ms 和 durations 属性。定义 current_ms 方法，用于返回当前时间的毫秒值。重写 on_chain_start、on_chain_end、on_llm_start 和 on_llm_end 方法，在这些方法中记录开始和结束时间，并计算处理耗时。接下来，我们创建了一个 OpenAI 实例、一个 TimerHandler 实例以及一个 PromptTemplate 实例。然后，我们创建了一个使用 timerHandler 作为回调处理器的 LLMChain 实例。最后，我们运行了两次Chain，分别查询蓝色和紫色的十六进制代码。在链运行过程中，TimerHandler 将记录每次交互的处理耗时，并将其添加到 durations 列表中。</font>

<font style="color:rgb(62, 71, 83);">输出如下：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1711029576510-e7535fe3-75c3-46d2-bc83-44568edaa964.png)

#### **<font style="color:rgb(0, 0, 0);">8.3callbacks使用场景总结</font>**
<font style="color:rgb(62, 71, 83);">1️⃣</font><font style="color:rgb(62, 71, 83);">通过构造函数参数 callbacks 设置。这种方式可以在创建对象时就设置好回调处理器。例如，在创建 LLMChain 或 OpenAI 对象时，可以通过 callbacks 参数设置回调处理器。</font>

```python
timerHandler = TimerHandler()
llm = OpenAI(callbacks=[timerHandler]) 
response = llm.predict("What is the HEX code of color BLACK?") print(response)
```

<font style="color:rgb(62, 71, 83);">在这里构建llm的时候我们就直接指定了构造函数。</font>

<font style="color:rgb(62, 71, 83);">2️⃣</font><font style="color:rgb(62, 71, 83);">通过运行时的函数调用。这种方式可以在运行时动态设置回调处理器，如在Langchain的各module如Model，Agent，Tool，以及 Chain的请求执行函数设置回调处理器。例如，在调用 LLMChain 的 run 方法或 OpenAI 的 predict 方法时，可以通过 callbacks 参数设置回调处理器。以OpenAI 的 predict 方法为例：</font>

```python
timerHandler = TimerHandler()
llm = OpenAI()
response = llm.predict("What is the HEX code of color BLACK?", callbacks=[timerHandler])
print(response)
```

<font style="color:rgb(62, 71, 83);">这段代码首先创建一个 TimerHandler 实例并将其赋值给变量 timerHandler。然后创建一个 OpenAI 实例并将其赋值给变量 llm。调用 llm.predict() 方法，传入问题 "What is the HEX code of color BLACK?"，并通过 callbacks 参数设置回调处理器 timerHandler。</font>

<font style="color:rgb(62, 71, 83);">两种方法的主要区别在于何时和如何设置回调处理器。</font>

<font style="color:rgb(62, 71, 83);">构造函数参数 callbacks 设置：在创建对象（如 OpenAI 或 LLMChain）时，就通过构造函数的 callbacks 参数设置回调处理器。这种方式的优点是你可以在对象创建时就确定回调处理器，后续在使用该对象时，无需再次设置。但如果在后续的使用过程中需要改变回调处理器，可能需要重新创建对象。</font>

<font style="color:rgb(62, 71, 83);">通过运行时的函数调用：在调用对象的某个方法（如 OpenAI 的 predict 方法或 LLMChain 的 run 方法）时，通过该方法的 callbacks 参数设置回调处理器。这种方式的优点是你可以在每次调用方法时动态地设置回调处理器，更加灵活。但每次调用方法时都需要设置，如果忘记设置可能会导致回调处理器不生效。</font>

<font style="color:rgb(62, 71, 83);">在实际使用中，可以根据需要选择合适的方式。如果回调处理器在对象的整个生命周期中都不会变，可以选择在构造函数中设置；如果回调处理器需要动态变化，可以选择在运行时的函数调用中设置。</font>

### **<font style="color:rgb(0, 0, 0);">九、总结</font>**
<font style="color:rgb(62, 71, 83);">至此，Langchain的各个模块使用方法就已经介绍完毕啦，相信你已经感受到Langchain的能力了～不难发现，LangChain 是一个功能十分强大的AI语言处理框架，它将Model IO、Retrieval、Memory、Chains、Agents和Callbacks这六个模块组合在一起。Model IO负责处理AI模型的输入和输出，Retrieval模块实现了与向量数据库相关的检索功能，Memory模块则负责在对话过程中存储和重新加载历史对话记录。Chains模块充当了一个连接器的角色，将前面提到的模块连接起来以实现更丰富的功能。Agents模块通过理解用户输入来自主调用相关工具，使得应用更加智能化。而Callbacks模块则提供了回调机制，方便开发者追踪调用链路和记录日志，以便更好地调试LLM模型。总之，LangChain是一个功能丰富、易于使用的AI语言处理框架，它可以帮助开发者快速搭建和优化AI应用。</font>

<font style="color:rgb(62, 71, 83);">本文只是列举了各模块的核心使用方法和一些示例demo，建议结合本文认真阅读一遍官方文档会更加有所受益～</font>

