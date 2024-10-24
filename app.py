
import streamlit as st
from PIL import Image
from utils import summarization, image_to_text, text_to_text

#########
# text-to-text
#########
st.header('文章要約')
input = st.text_input("テキストを入力してください")
if st.button("送信"):
    result = text_to_text(input)
    st.text_area("出力結果", result)

#########
# summarization
#########
st.header('テキストファイルを読み込んで要約')
#モデルの選択
st.text('使用するモデルを選択してください')
reduce_llm_key = st.radio(
    'ChatGPTのモデル', 
    ['GPT-3.5-turbo', 'GPT-4o']
)

# ファイルのアップロード
uploaded_file = st.file_uploader("ファイルを選択してください")
if uploaded_file is not None:
    # ファイルの読み込み
    text = uploaded_file.read().decode("utf-8")

    # プロンプト
    map_prompt_template = """内容を要約してください。
    ------
    {text}
    ------
    """

    map_combine_template = """内容をまとめて要約してください。
    ------
    {text}
    ------
    """
    # モデルを選択
    reduce_llm_dic = {'GPT-3.5-turbo': "gpt-3.5-turbo", 'GPT-4o': "gpt-4o"}
    reduce_llm = reduce_llm_dic[reduce_llm_key]
    llm = "gpt-3.5-turbo"
    result = summarization(llm, reduce_llm, map_prompt_template, map_combine_template, text)
    # 出力欄に結果を表示
    st.text_area("出力結果", result)

# 画像入力
st.header('画像を読み込んで大喜利')
file_path = st.file_uploader('画像をアップロードしてください', type=['png', 'jpg', 'jpeg'])
if file_path is not None:
    image = Image.open(file_path)
    result = image_to_text(image)
    st.text_area("出力結果", result)