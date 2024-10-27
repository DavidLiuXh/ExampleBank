import openai
import numpy as np

# 设置API密钥
key = '您的OpenAI API Key'

client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=key,
)

# example1
COMPLETION_MODEL = "gpt-3.5-turbo"

def example1():
    current_messages = [
            {
                "role": "user",
                "content": "Say this is a test",
                }
            ]

    completion = client.chat.completions.create(
            messages=current_messages,
            model=COMPLETION_MODEL)
    # 输出回复内容

    print(completion)
    print(completion.choices[0].message.content)

# example2
COMPLETION_MODEL = "gpt-3.5-turbo"
def get_embedding(text):
    global EMBEDDING_MODEL

    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=EMBEDDING_MODEL).data[0].embedding

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    epsilon = 1e-10
    cosine_similarity = dot_product / (norm_a * norm_b + epsilon)
    return cosine_similarity

def check_positive(evaluate):
    global EMBEDDING_MODEL
    EMBEDDING_MODEL = "text-embedding-ada-002"

    positive_review = get_embedding("好评")
    negative_review = get_embedding("差评")

    current_review = get_embedding(evaluate)

    similarity = cosine_similarity(current_review, positive_review) - cosine_similarity(current_review, negative_review)
    if similarity > 0:
        print("当前的评价是正面的。")
    else:
        print("当前的评价是负面的。")

# 新闻分类
def check_news_category(evaluate):
    global EMBEDDING_MODEL
    EMBEDDING_MODEL = "text-embedding-ada-002"

    technology_news = get_embedding("科技新闻")
    sport_news = get_embedding("体育新闻")
    recreation_news = get_embedding("娱乐新闻")

    current_review = get_embedding(evaluate)

    technology_similarity = cosine_similarity(current_review, technology_news)
    sport_similarity = cosine_similarity(current_review, sport_news)
    recreation_similarity = cosine_similarity(current_review, recreation_news)
    
    most_match_category = max(technology_similarity, sport_similarity, recreation_similarity)
    if most_match_category == technology_similarity:
        print("科技新闻")
    elif most_match_category == sport_similarity:
        print("体育新闻")
    else:
        print("娱乐新闻")

#example1()
#check_positive("随意降价，不予价保，服务态度差")
#check_positive("质量不算坏，价格不算高")
check_news_category("17岁105天，亚马尔成为在西甲对阵皇马进球最年轻球员")

