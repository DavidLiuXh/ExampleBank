<font style="color:rgb(0, 0, 0);">如果你对prompt还不是很了解，推荐阅读这个prompt调查报告。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1719111225109-7d30e995-09e8-41d5-9fc8-65c2a55bf926.png)

<font style="color:rgb(0, 0, 0);">这篇报告由马里兰大学的研究团队主要撰写，此外，报告的作者还包括来自</font>OpenAI<font style="color:rgb(0, 0, 0);">、斯坦福大学、微软、普林斯顿大学等知名机构的研究人员，共有38位作者。从作者的学术背景来看，这个研究团队由自然语言处理、机器学习等领域的专家和博士生组成，主要来自马里兰大学这一NLP研究的重镇，同时也有来自工业界巨头的成员。这样的团队组成应该能够在一定程度上保证报告的学术质量和权威性。</font>

## <font style="color:rgb(0, 0, 0);">Prompt的沿革和定义</font>
<font style="color:rgb(0, 0, 0);">使用自然语言前缀(prefix)或提示(prompt)来引导语言模型行为和输出的想法早在GPT-3和</font>ChatGPT<font style="color:rgb(0, 0, 0);">时代之前就已出现。2018年，Fan等人首次在</font>生成式AI<font style="color:rgb(0, 0, 0);">的语境中使用了prompt。此后，Radford等人在2019年发布的GPT-2中也采用了prompt。</font>

<font style="color:rgb(0, 0, 0);">不过，prompt的概念可以追溯到更早的一些相关概念，如控制码(control code)和写作提示(writing prompt)。而"prompt engineering"这一术语则是在2021年前后由Radford等人和Reynolds & McDonell等人提出的。</font>

<font style="color:rgb(0, 0, 0);">有趣的是，早期对prompt的定义与当前普遍理解略有不同。比如在2020年的一项工作中，Brown等人给出了这样一个prompt示例："Translate English to French：llama"，他们认为其中的"llama"才是真正的prompt，而"Translate English to French："是一个"任务描述"。相比之下，包括本文在内的大多数近期工作都将输入给语言模型的整个字符串视为prompt。</font>

<font style="color:rgb(0, 0, 0);">利用prompt来驾驭语言模型完成各类任务的技术范式经历了从概念萌芽到逐步成熟的过程。随着GPT-3等大规模语言模型的问世，prompt engineering迅速成为了自然语言处理领域的研究热点，并衍生出诸多创新方法。</font>

<font style="color:rgb(0, 0, 0);">报告首先厘清了Prompt的定义。综合已有的表述，报告提出了一个简洁而全面的定义：Prompt是一段输入给生成式AI模型的文本，用于引导模型产生期望的输出。这个定义点明了Prompt的本质属性和功能定位。</font>

## <font style="color:rgb(0, 0, 0);">Prompt的六大构成要素</font>
1. <font style="color:rgb(1, 1, 1);">指令(Directive)：这是Prompt的灵魂所在。通过精心设计的指令，我们可以向模型传达任务的核心诉求。举个例子，如果我们想要生成一篇关于春天的诗歌，可以使用"请写一首歌颂春天美好的诗"这样的指令。指令的表述要明确、具体，避免歧义。</font>
2. <font style="color:rgb(1, 1, 1);">示例(Example)：这是In-Context Learning的关键。通过在Prompt中提供几个精选的示例，我们可以让模型快速理解任务的输入输出格式和要求。比如，在情感分类任务中，我们可以提供几个样本文本及其情感标签(正面/负面)，让模型学会判断情感倾向。示例要典型、多样，覆盖任务的主要场景。</font>
3. <font style="color:rgb(1, 1, 1);">格式控制(Output Formatting)：这是规范模型输出的利器。通过格式控制标记，我们可以让模型以特定的格式组织输出内容，如生成CSV格式的表格、Markdown格式的文档等。例如，在数据分析任务中，我们可以要求模型以表格形式输出统计结果，每一行对应一个指标，用逗号分隔。</font>
4. <font style="color:rgb(1, 1, 1);">角色指定(Role)：这是激发模型创造力的神奇钥匙。通过为模型赋予一个虚拟的身份，我们可以让它以特定的视角、风格生成内容。比如，我们可以让模型扮演一位历史学家，以严谨的笔调评述一段历史事件;也可以让它化身为一名诗人，用优美的语言描绘大自然的风光。</font>
5. <font style="color:rgb(1, 1, 1);">风格指令(Style Instruction)：这是调控模型语言风格的调色板。通过风格指令，我们可以要求模型以特定的语气、情感倾向、字数限制等生成内容。例如，我们可以指示模型用严肃的口吻撰写一份商业报告，或是用幽默风趣的笔调创作一个段子。</font>
6. <font style="color:rgb(1, 1, 1);">补充信息(</font>Additional Information<font style="color:rgb(1, 1, 1);">)：这是为模型提供背景知识的补给站。很多任务需要一定的领域知识作为辅助信息。比如，在撰写一篇医学论文时，我们可以为模型提供一些疾病的定义、治疗方案等背景资料，帮助模型更好地理解和表述主题。</font>

## <font style="color:rgb(0, 0, 0);">26 种 prompt 套路</font>
<font style="color:rgb(0, 0, 0);">此外，还有大佬整理了26 种 prompt 套路，觉得挺有用，以此分享给大家。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/35727243/1719111225008-d11bc3b6-350d-4d50-9fc7-43401c369657.png)

<font style="color:rgb(136, 136, 136);">论文地址：  
</font><font style="color:rgb(136, 136, 136);">https://arxiv.org/abs/2312.16171</font>

<font style="color:rgb(136, 136, 136);">相关代码：  
</font><font style="color:rgb(136, 136, 136);">https://github.com/VILA-Lab/ATLAs</font><font style="color:rgb(0, 0, 0);">  
</font>

1. <font style="color:rgb(1, 1, 1);">如果你想要简洁的回答，不用太客气，直接说就行，不用加上“请”、“如果你不介意”、“谢谢”、“我想要”等客套话。</font>
2. <font style="color:rgb(1, 1, 1);">在提问时说明目标受众，例如，告诉 LLM 你的受众是该领域的专家。</font>
3. <font style="color:rgb(1, 1, 1);">把复杂的任务分成几个简单的小问题，逐步解决。</font>
4. <font style="color:rgb(1, 1, 1);">用肯定的语气说“做某事”，避免用否定语气说“不要做某事”。</font>
5. <font style="color:rgb(1, 1, 1);">当你需要更清楚或深入了解某个话题时，可以这样提问：</font>
    - <font style="color:rgb(1, 1, 1);">用简单的语言解释[具体话题]。</font>
    - <font style="color:rgb(1, 1, 1);">向我解释，就像我 11 岁一样。</font>
    - <font style="color:rgb(1, 1, 1);">向我解释，就像我是[领域]的新手一样。</font>
    - <font style="color:rgb(1, 1, 1);">用简单的英文写[文章/文本/段落]，就像你在向 5 岁的小孩解释。</font>
6. <font style="color:rgb(1, 1, 1);">加上“如果有更好的解决方案，我会奖励 xxx”。</font>
7. <font style="color:rgb(1, 1, 1);">用具体的例子来提问（即使用几个示例来引导）。</font>
8. <font style="color:rgb(1, 1, 1);">在你的提问前写上“###指示###”，如果相关的话，再加上“###示例###”或“###问题###”，然后再写你的内容。用空行分隔指示、示例、问题、背景和输入数据。</font>
9. <font style="color:rgb(1, 1, 1);">使用“你的任务是”和“你必须”这样的短语。</font>
10. <font style="color:rgb(1, 1, 1);">使用“你将受到惩罚”这样的短语。</font>
11. <font style="color:rgb(1, 1, 1);">使用“像人一样自然地回答问题”这样的短语。</font>
12. <font style="color:rgb(1, 1, 1);">用引导词，比如“一步步来思考”。</font>
13. <font style="color:rgb(1, 1, 1);">在提问中加上“确保你的回答没有偏见，避免刻板印象”。</font>
14. <font style="color:rgb(1, 1, 1);">让 LLM 向你提问，直到它有足够的信息来回答你。例如，“从现在起，请你问我问题，直到你有足够的信息……”。</font>
15. <font style="color:rgb(1, 1, 1);">如果你想测试对某个话题的理解，可以这样说：“教我[定理/话题/规则]，最后加个测试，等我回答后告诉我是否正确，但不要提前给答案。”</font>
16. <font style="color:rgb(1, 1, 1);">给 LLM 指定一个角色。</font>
17. <font style="color:rgb(1, 1, 1);">使用分隔符。</font>
18. <font style="color:rgb(1, 1, 1);">在提问中多次重复某个特定的词或短语。</font>
19. <font style="color:rgb(1, 1, 1);">将链式思维（</font>CoT<font style="color:rgb(1, 1, 1);">）和少量示例的提示结合使用。</font>
20. <font style="color:rgb(1, 1, 1);">使用输出引导语，在你的提问结尾加上预期回答的开头部分。</font>
21. <font style="color:rgb(1, 1, 1);">想写详细的文章、段落或文本时，可以这样说：“请为我写一篇详细的[文章/段落]，内容涉及[话题]，并加入所有必要的信息。”</font>
22. <font style="color:rgb(1, 1, 1);">如果你要修改特定文本但不改变风格，可以这样说：“请修改用户发送的每个段落，只需改进语法和词汇，使其听起来自然，但保持原有的写作风格，确保正式的段落仍然正式。”</font>
23. <font style="color:rgb(1, 1, 1);">当你有复杂的代码提示需要分成不同文件时，可以这样说：“从现在起，每当你生成跨多个文件的代码时，生成一个[编程语言]脚本，以自动创建指定的文件或修改现有文件以插入生成的代码。”然后提问。</font>
24. <font style="color:rgb(1, 1, 1);">当你想用特定的词、短语或句子来开始或继续一段文字时，可以使用以下提示：“我提供给你开头部分[歌词/故事/段落/文章...]: [插入歌词/词语/句子]。请根据提供的词语完成它，并保持一致的流畅性。”</font>
25. <font style="color:rgb(1, 1, 1);">明确指出模型必须遵循的要求，以关键词、规则、提示或指令的形式。</font>
26. <font style="color:rgb(1, 1, 1);">想写与提供的样本相似的文本时，可以这样说：“请根据提供的段落[/标题/文本/文章/答案]使用相同的语言。”</font>

