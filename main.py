# import lib
import os
import re
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.openai import OpenAIEmbeddings
from openai import AsyncOpenAI
import asyncio
from streamlit_star_rating import st_star_rating
import pandas as pd
import matplotlib.pyplot as plt
from retriever import process_query  # Import query expansion module

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'llm-chatbot-gpt'

primer2 = """Bây giờ bạn hãy đóng vai trò là một luật sư xuất sắc về luật hôn nhân và gia đình ở Việt Nam.
Tôi sẽ hỏi bạn các câu hỏi về tình huống thực tế liên quan tới luật hôn nhân và gia đình. Bạn hãy tóm tắt tình huống
và đưa ra các câu hỏi ngắn gọn gồm từ khoá liên quan tới pháp luật được suy luận từ phần thông tin có trong tình huống. Các câu trả lời của bạn
đều là tiếng việt."""

primer1 = """Bây giờ bạn hãy đóng vai trò là một luật sư xuất sắc về luật hôn nhân và gia đình ở Việt Nam.
Tôi sẽ hỏi bạn các câu hỏi về tình huống thực tế liên quan tới luật hôn nhân và gia đình. Bạn sẽ trả lời câu hỏi
dựa trên thông tin tôi cung cấp và thông tin có trong câu hỏi. Nếu thông tin tôi cung cấp không đủ để trả lời
hãy nói rằng 'Tôi không biết'. Các câu trả lời của bạn đều là tiếng việt. 
Lưu ý: Nêu rõ điều luật số mấy để trả lời tình huống.
"""


def check_string(s):
    pattern = r"Nếu bạn cần"
    return re.search(pattern, s) is not None


def remove_string(s):
    pattern = r"Nếu bạn cần .*"
    return re.sub(pattern, "", s)


async def qa1(prompt, client):
    """
    Get the primary answer from GPT without additional context retrieval.
    """
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer1},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


async def qa2(prompt, user_openai_api_key):
    """
    Expand query, retrieve documents, and generate an enhanced response.
    Include expanded terms and retrieved contexts for transparency.
    """
    query_data = process_query(prompt, user_openai_api_key=user_openai_api_key, k=3)
    expanded_terms = query_data["expanded_terms"]
    documents = query_data["documents"]

    context = "\n".join(doc.page_content for doc in documents) + "\n" + prompt

    client = AsyncOpenAI(api_key=user_openai_api_key)
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer1},
            {"role": "user", "content": context}
        ]
    )
    answer = response.choices[0].message.content.strip()

    if check_string(answer):
        answer = remove_string(answer)

    return answer, documents, expanded_terms


async def page_1():
    if "openai_apikey" not in st.session_state or not st.session_state.openai_apikey:
        st.error("API Key is missing. Please update your API Key in the 'Update API KEY' section.")
        return

    client = AsyncOpenAI(api_key=st.session_state.openai_apikey)

    st.title("🧑‍💻💬 A RAG chatbot for family and marriage legal questions")
    """
    Đây là chatbot giúp người dân tìm hiểu luật hôn nhân và gia đình. Bạn hãy hỏi những câu hỏi có liên quan tới luật này nhé.
    """

    for conversation in st.session_state.chat_history:
        st.chat_message("user").write(conversation['question'])
        col1, col2 = st.columns(2)
        with col1:
            st.chat_message("assistant").write(conversation['answer1'])
        with col2:
            st.chat_message("assistant").write(conversation['answer2'])

        with st.expander("Expanded Terms"):
            st.write(", ".join(conversation['expanded_terms']))

        with st.expander("Retrieved Contexts"):
            for i, doc in enumerate(conversation['retrieved_contexts']):
                st.write(f"**Context {i + 1}:** {doc.page_content}")

        st.write(f"You rated this answer {conversation['stars']} :star:")

    if st.session_state.get("stars"):
        df = pd.read_csv('result.csv')
        new_row = {
            'question': st.session_state['question'],
            'gpt_answer': st.session_state['msg1'],
            'enhanced_answer': st.session_state['msg2'],
            'rating': st.session_state['stars']
        }
        df = df._append(new_row, ignore_index=True)
        df.to_csv('./result.csv', index=False)

    if prompt := st.chat_input():
        st.session_state.prompt = prompt
        st.chat_message("user").write(prompt)
        msg1 = await qa1(prompt, client)
        msg2, retrieved_contexts, expanded_terms = await qa2(prompt, st.session_state.openai_apikey)

        col1, col2 = st.columns(2)
        with col1:
            st.chat_message("assistant").write(msg1)
        with col2:
            st.chat_message("assistant").write(msg2)

        st.session_state['retrieved_contexts'] = retrieved_contexts
        st.session_state['expanded_terms'] = expanded_terms
        st.session_state['question'] = prompt
        st.session_state['msg1'] = msg1
        st.session_state['msg2'] = msg2

    if "prompt" in st.session_state:
        st_star_rating("Please rate your experience", maxValue=4, defaultValue=3, key="stars")


async def page_2():
    df = pd.read_csv('result.csv')
    st.write("Data:")
    st.write(df)
    st.write(f"There is a total of {len(df)} answered questions")

    st.write("Histogram of 'rating' column:")
    fig, ax = plt.subplots()
    ax.hist(df['rating'], bins=4)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['1', '2', '3', '4'])

    st.pyplot(fig)


def page_3():
    st.title("Nhập API Key")
    api_key_input = st.text_input("API Key:", type="password")
    if st.button("Lưu API Key"):
        st.session_state.openai_apikey = api_key_input
        st.success("API Key đã được lưu!")


PAGES = {
    "Chat": page_1,
    "Statistic": page_2,
    "Update API KEY": page_3
}

def main():
    st.set_page_config(page_title="RAG Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "openai_apikey" not in st.session_state:
        st.session_state.openai_apikey = None

    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Select an option", list(PAGES.keys()))

    if choice != "Update API KEY":
        asyncio.run(PAGES[choice]())
    else:
        PAGES[choice]()


if __name__ == "__main__":
    main()
