# -*- coding: UTF-8 -*-

import openai

# 设置API密钥
key = '[]'

COMPLETION_MODEL = "gpt-3.5-turbo"

class ChatRobot:
    def __init__(self, num_of_round):
        self.num_of_round = num_of_round
        if self.num_of_round < 4:
            self.num_of_round = 4

        self.chat_history = []
        self.client = openai.OpenAI(api_key=key,)

    def set_chat_category(self, category):
        self.chat_history = []

        # set system role
        system_msg = category + """请用你的专业知识尽全力为我解答问题。
        在回答问题前请先对这个问题作一个评价,然后再开始正式的回答,遵循下面的形式：
        1. 评价：xxx;
        2. 答案：xxxx。"""
        self.chat_history.append(
                { "role": "system", "content": system_msg, })

    def set_chat_category_by_few_shot(self, category):
        self.chat_history = []

        # set system role
        system_msg = """请评估我所提问题的常见性，根据结果回答'常见'或'不常见'。"""
        self.chat_history.append(
                { "role": "system", "content": system_msg, })

        # set few shot by user role and assistant role
        self.chat_history.append(
                { "role": "user", "content": """ 请问如何缓解焦虑？ """, })
        self.chat_history.append(
                { "role": "assistant", "content": """常见""", })

        self.chat_history.append(
                { "role": "user", "content": """ 为什么大便可以缓解肚子疼？""", })
        self.chat_history.append(
                { "role": "assistant", "content": """不常见""", })

    def ask(self, question):
        if question == "":
            print("The question cannot be empty")
            return

        self.chat_history.append(
                {
                    "role": "user", "content": question,
                    })
        try:
            completion = self.client.chat.completions.create(
                    model=COMPLETION_MODEL,
                    messages=self.chat_history,
                    temperature=0.5,
                    max_tokens=2048,
                    top_p=1,)
        except Exception as e:
            print("Fail to gpt: %s", e)
            return

        message = completion.choices[0].message.content
        self.chat_history.append(
                {"role": "assistant", "content": message})

        if len(self.chat_history) > self.num_of_round * 2 + 1:
            del self.chat_history[5:7]

        return message

if __name__ == "__main__":
    chat_robot = ChatRobot(4)
    chat_robot.set_chat_category("假定你是人体生理医学方面的专家。")
    
    print("您可以询问一些人体生理医学方面的问题，且回复仅供参考。\n\n")

    problem = input("User: ")
    while problem != "quit":
        if problem == "change":
            categary = input("更改问题类型: ")
            #chat_robot.set_chat_category(categary)
            chat_robot.set_chat_category_by_few_shot(categary)
        else:
            answer = chat_robot.ask(problem)
            print("Assistent: {}\n".format(answer))

        problem = input("User: ")
