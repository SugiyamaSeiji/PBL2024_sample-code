from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import base64
import io
import os

os.environ["OPENAI_API_KEY"] = ""

#########
# text-to-text
#########
def text_to_text(text):
    with open("input.txt", "a") as f:
            f.write(text)

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {
            "role": "system",
            "content": "あなたは、要約者です。",
        },
        {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"次の文章を500字程度に要約してください。「{text}」"},
                ],
            }
        ]
    )
    response_text = response.choices[0].message.content
    with open("output.txt", "a") as f:
        f.write(response_text)
    return response_text

#########
# summarization
#########
def summarization(llm, reduce_llm, map_prompt_template, map_combine_template, text):
    with open("input.txt", "a") as f:
        f.write(text)

    # テキストを分割する
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 5000,   # チャンクの文字数
        chunk_overlap  = 0,  # チャンクオーバーラップの文字数  
    )

    text_splitted = text_splitter.split_text(text)

    docs = [Document(page_content=i) for i in text_splitted] # Documentオブジェクト化

    # map_reduce法による要約を作成
    map_first_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    map_combine_prompt = PromptTemplate(template=map_combine_template, input_variables=["text"])

    map_chain = load_summarize_chain(
        llm = ChatOpenAI(temperature=0,model_name=llm), # 分割ドキュメントを要約
        reduce_llm = ChatOpenAI(temperature=0,model_name=reduce_llm), # 最終の要約モデル
        collapse_llm = ChatOpenAI(temperature=0,model_name=reduce_llm), # 入力制限時に使用される
        chain_type = "map_reduce",
        map_prompt = map_first_prompt,
        combine_prompt = map_combine_prompt,
        collapse_prompt = map_combine_prompt,
        token_max = 5000,
        verbose = True
    )

    result = map_chain({"input_documents": docs}, return_only_outputs=False) 
    with open("output.txt", "a") as f:
        f.write(result["output_text"])

    return result["output_text"]

###############
# Image-to-Text
###############

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format=image.format or "PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def image_to_text(image):
    # `encode_image_to_base64`で PIL 形式の画像を base64 エンコードした文字列に変換
    image_base64 = encode_image_to_base64(image)
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
        {
            "role": "system",
            "content": "あなたはユーモアに溢れた大喜利 AI です。与えられたお題に対して、面白い回答をお願いします。",
        },
        {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "次の画像を見て、何か面白い一言をお願いします。",
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{image_base64}",
                    "detail": "low"
                },
                },
            ],
        }
        ]
    )
    response_text = response.choices[0].message.content
    with open("output.txt", "a") as f:
        f.write(response_text)

    
    return response_text
