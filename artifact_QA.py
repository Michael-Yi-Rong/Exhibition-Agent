import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import ast
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Step 1: Load the precomputed feature dataset (feature_offline)
feature_offline_path = r'E:\认知智能\Final_Codes\datasets\exhibition_feature_offline.csv'
triplet_path = r'E:\认知智能\Final_Codes\datasets\exhibition_combined.csv'

feature_offline = pd.read_csv(feature_offline_path)

def preprocess_features(feature_offline):
    processed_features = []

    for _, row in feature_offline.iterrows():
        # 获取 image_features 列的字符串
        feature_str = row['image_features']

        # 如果是 NaN 或者为空字符串，跳过
        if pd.isna(feature_str) or feature_str.strip() == "":
            processed_features.append((row['systemNumber'], torch.zeros(512)))
            continue

        try:
            # 去掉方括号并将字符串转换为浮点数列表
            feature_str = feature_str.strip("[]")
            feature_list = list(map(float, feature_str.split()))  # 将空格分隔的字符串转换为浮点数列表

            # 转换为 PyTorch tensor
            feature_tensor = torch.tensor(feature_list, dtype=torch.float32)

            # 如果特征向量全为零，避免除零错误，直接跳过
            norm = feature_tensor.norm()
            if norm != 0:
                feature_tensor = feature_tensor / norm  # 归一化
            else:
                feature_tensor = torch.zeros_like(feature_tensor)  # 如果全为零，返回全零 tensor

            processed_features.append((row['systemNumber'], feature_tensor))

        except Exception as e:
            # 如果出现异常，返回全零向量
            print(f"Error processing feature for systemNumber {row['systemNumber']}: {e}")
            processed_features.append((row['systemNumber'], torch.zeros(512)))

    return processed_features
processed_features= preprocess_features(feature_offline)

triplet = pd.read_csv(triplet_path, encoding='ISO-8859-1')
standard_queries={"What is the type of this artifact?":"objectType",
                  "What is the weight of this artifact?":"Weight",
                  "What is the width of this artifact?": "Width",
                  "What is the maker's name of this artifact?": "_primaryMaker_name",
                  "Where is this artifact on display now?": "_currentLocation_displayName",
                  "Where is the origin of this artifact": "_primaryPlace",
                  "What is the age of this artifact?":"_primaryDate",
                  "What is the title of this artifact":"_primaryTitle",
                  "What is the depth of this artifact?":"Depth",
                  "What is the Diameter of this artifact?":"Diameter",
                  "What is the Height of this artifact?":"Height",
                  "What is the Length of this artifact?":"Length",
                  "What is the materials and techniques of this artifact?":"materialsAndTechniques",

                  }

# Step 2: Load the CLIP model and processor
clip_path=r'D:\models\clip'
clip_model = CLIPModel.from_pretrained(clip_path)
clip_processor = CLIPProcessor.from_pretrained(clip_path)
# 加载 BERT Tokenizer 和模型
bert_path=r'D:\models\bert'
tokenizer = BertTokenizer.from_pretrained(bert_path)
model = BertModel.from_pretrained(bert_path)

# Step 3: Define a function to compute the similarity and find the closest match

def find_closest_systemnumber(query_image_path, processed_features):
    # 加载和预处理查询图片
    query_image = Image.open(query_image_path).convert("RGB")
    inputs = clip_processor(images=query_image, return_tensors="pt")

    # 获取查询图片的嵌入
    with torch.no_grad():
        query_embedding = clip_model.get_image_features(**inputs)

    # 归一化查询嵌入
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

    # 计算与离线特征的余弦相似度
    similarities = []
    for system_number, offline_feature in processed_features:
        similarity = torch.dot(query_embedding.squeeze(), offline_feature)
        similarities.append((system_number, similarity.item()))

    # 找到相似度最高的 systemNumber
    closest_systemnumber = max(similarities, key=lambda x: x[1])[0]
    return closest_systemnumber

# Step 5: Define a function to extract the feature column based on user query
# 函数：将文本转换为嵌入（向量）
def get_embedding(text):
    if isinstance(text, str):  # 确保输入是字符串
        # 将文本编码为模型可以接受的输入
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # 获取模型输出（最后一层隐藏状态）
        with torch.no_grad():  # 不需要计算梯度
            outputs = model(**inputs)

        # 提取最后一层隐藏状态的平均池化表示作为文本的嵌入
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()  # squeeze去掉多余的维度

        return embeddings
    else:
        # 将输入强制转换为字符串
        text = str(text)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # 获取模型输出（最后一层隐藏状态）
        with torch.no_grad():  # 不需要计算梯度
            outputs = model(**inputs)

        # 提取最后一层隐藏状态的平均池化表示作为文本的嵌入
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()  # squeeze去掉多余的维度

        return embeddings


# 计算最匹配的特征

def get_target_feature(query_text, standard_queries_dict):
    # 获取查询的嵌入
    query_embedding = get_embedding(query_text)

    # 初始化相似度字典
    similarities = {}

    # 遍历标准查询字典，计算与每个标准查询的相似度
    for standard_query, feature in standard_queries_dict.items():
        # 获取标准查询的嵌入
        standard_query_embedding = get_embedding(standard_query)

        # 计算余弦相似度
        similarity = cosine_similarity(query_embedding.unsqueeze(0), standard_query_embedding.unsqueeze(0))[0][0]

        # 保存相似度和对应的特征名
        similarities[standard_query] = (similarity, feature)

    # 找到相似度最大的标准查询
    most_similar_query = max(similarities, key=lambda x: similarities[x][0])

    # 返回最相似查询对应的特征名
    return similarities[most_similar_query][1]

# Step 6: Retrieve the value of the target feature from the triplet dataset
def get_feature_value(systemnumber, target_feature):
    # 过滤 triplet 数据集，查找匹配的 systemNumber 和 target_feature
    result = triplet[(triplet['systemNumber'] == systemnumber) & (triplet['Columns'] == target_feature)]

    # 如果找到匹配的结果
    if not result.empty:
        value = result.iloc[0]['Value']  # 获取匹配行的 Value 列的值
        return value
    else:
        return None  # 如果没有匹配的结果，返回 None


# Step 7: Integrate the pipeline
def artifact_query_pipeline(query_image_path, query_text):
    # Step 7.1: Find the closest systemnumber
    closest_systemnumber = find_closest_systemnumber(query_image_path,processed_features)

    # Step 7.2: Determine the target feature from the query
    target_feature = get_target_feature(query_text,standard_queries)

    # Step 7.3: Retrieve the feature value from the triplet dataset
    feature_value = get_feature_value(closest_systemnumber, target_feature)
    print(target_feature,closest_systemnumber,feature_value)
    if feature_value:
        return f"The {target_feature} of this artifact is {feature_value}."
    else:
        return "Sorry, the required information could not be found."

# Example usage
query_image_path = r'E:\认知智能\Final_Codes\downloaded_images\O151308.jpg'
query_text = "How long is this artifact?"
response = artifact_query_pipeline(query_image_path, query_text)
print(response)
