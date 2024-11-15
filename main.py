# import lib
import os
import time
import re
import openai
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.openai import OpenAIEmbeddings
from openai import AsyncOpenAI
import asyncio
from streamlit_star_rating import st_star_rating
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from retriever import process_query  # Import query expansion module

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'llm-chatbot-gpt'
embedding = None
client = None
pVS = None


primer2 = f"""Bây giờ bạn hãy đóng vai trò là một luật sư xuất sắc về luật hôn nhân và gia đình ở Việt Nam.
Tôi sẽ hỏi bạn các câu hỏi về tình huống thực tế liên quan tới luật hôn nhân và gia đình. Bạn hãy tóm tắt tình huống
và đưa ra các câu hỏi ngắn gọn gồm từ khoá liên quan tới pháp luật được suy luận từ phần thông tin có trong tình huống. Các câu trả lời của bạn
đều là tiếng việt.

Sau đây là một số ví dụ và phần tóm tắt:
1. Tình huống: Biết mình đủ tuổi kết hôn và đáp ứng các điều kiện kết hôn, Anh S và chị Y dự định đi đăng ký kết hôn trước khi tổ chức lễ cưới 02 tháng. Chị Y và anh S có hộ khẩu thường trú ở hai tỉnh khác nhau, anh chị muốn biết việc đăng ký kết hôn thực hiện tại cơ quan nào và cần thực hiện thủ tục gì?
-> Tóm tắt: Thủ tục đăng ký kết hôn là gì, hộ khẩu thường trú trong thủ tục kết hôn

2. Ông bà B có con trai đã 25 tuổi, bị bệnh đao bẩm sinh. Vì muốn lấy vợ cho con trai, bà B đã tìm cách vu cáo cho chị Y – người giúp việc lấy trộm số tiền 1.000.000 đồng. Bà B  đe dọa nếu chị Y không muốn bị báo công an, không muốn bị đi tù thì phải lấy con trai bà, vừa được làm chủ nhà, không phải làm người giúp việc lại có cuộc sống sung túc. Vì nhận thức hạn chế, trình độ văn hóa thấp nên chị Y đã đồng ý lấy con trai bà B. Hôn lễ chỉ tổ chức giữa hai gia đình mà không làm thủ tục đăng ký kết hôn tại phường. Việc làm của bà B có vi phạm pháp luật không? Nếu có thì bị xử phạt như thế nào?
-> Tóm tắt: cưỡng ép kết hôn có bị xử phạt không, cưỡng ép kết hôn bị xử phạt như thế nào 

3. Tôi đã kết hôn được 6 tháng, nhưng chưa chuyển hộ khẩu về nhà chồng (ở xã X, huyện B, tỉnh A), hộ khẩu của tôi vẫn đang ở nhà bố mẹ đẻ (xã Y, huyện C, tỉnh D). Nay tôi có nguyện vọng chuyển hộ khẩu về nhà chồng thì có được không và thủ tục thực hiện như thế nào? Ai có thẩm quyền giải quyết?
-> tóm tắt: có được chuyển hộ khẩu về nhà chồng không, Thủ tục chuyển hộ khẩu, Ai giải quyết thủ tục chuyển hộ khẩu
"""

primer1 = f"""Bây giờ bạn hãy đóng vai trò là một luật sư xuất sắc về luật hôn nhân và gia đình ở Việt Nam.
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


async def qa1(prompt):
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


async def qa2(prompt):
    """
    Expand query, retrieve documents, and generate an enhanced response.
    Include expanded terms and retrieved contexts for transparency.
    """
    # Expand query and retrieve documents
    query_data = process_query(prompt, k=3)
    expanded_terms = query_data["expanded_terms"]  # List of expanded terms
    documents = query_data["documents"]           # List of retrieved documents

    # Combine retrieved content
    context = "\n".join(doc.page_content for doc in documents) + "\n" + prompt

    # Get the enhanced answer from GPT
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

    return answer, documents, expanded_terms  # Return the answer, contexts, and expanded terms


async def page_1():
    global client
    client = AsyncOpenAI(api_key=st.session_state.openai_apikey)

    st.title("🧑‍💻💬 A RAG chatbot for family and marriage legal questions")
    """
    Đây là chatbot giúp người dân tìm hiểu luật hôn nhân và gia đình. Bạn hãy hỏi những câu hỏi có liên quan tới luật này nhé.
    """

    # Display chat history
    for conversation in st.session_state.chat_history:
        st.chat_message("user").write(conversation['question'])
        col1, col2 = st.columns(2)
        with col1:
            st.chat_message("assisstant").write(conversation['answer1'])
        with col2:
            st.chat_message("assisstant").write(conversation['answer2'])

        # Box to display expanded terms
        with st.expander("Expanded Terms"):
            st.write(", ".join(conversation['expanded_terms']))

        # Box to display retrieved contexts
        with st.expander("Retrieved Contexts"):
            for i, doc in enumerate(conversation['retrieved_contexts']):
                st.write(f"**Context {i + 1}:** {doc.page_content}")

        st.write(f"You rated this answer {conversation['stars']} :star:")

    if "stars" not in st.session_state:
        st.session_state.stars = ""

    if st.session_state.stars:
        # Store feedback
        st.session_state.chat_history.append(
            {
                'question': st.session_state['question'],
                'answer1': st.session_state['msg1'],
                'answer2': st.session_state['msg2'],
                'expanded_terms': st.session_state['expanded_terms'],
                'retrieved_contexts': st.session_state['retrieved_contexts'],
                'stars': st.session_state['stars']
            }
        )
        df = pd.read_csv('result.csv')
        new_row = {
            'question': st.session_state['question'],
            'gpt_answer': st.session_state['msg1'],
            'enhanced_answer': st.session_state['msg2'],
            'rating': st.session_state['stars']
        }
        df = df._append(new_row, ignore_index=True)
        df.to_csv('./result.csv', index=False)

        st.chat_message("user").write(st.session_state['question'])
        col1, col2 = st.columns(2)
        with col1:
            st.chat_message("assisstant").write(st.session_state['msg1'])
        with col2:
            st.chat_message("assisstant").write(st.session_state['msg2'])

        with st.expander("Expanded Terms"):
            st.write(", ".join(st.session_state['expanded_terms']))

        with st.expander("Retrieved Contexts"):
            for i, doc in enumerate(st.session_state['retrieved_contexts']):
                st.write(f"**Context {i + 1}:** {doc.page_content}")

        st.write(f"You rated this answer {st.session_state['stars']} :star:")
        del st.session_state.prompt

    if prompt := st.chat_input():
        st.session_state.prompt = prompt
        st.chat_message("user").write(prompt)
        msg1 = await qa1(prompt)
        msg2, retrieved_contexts, expanded_terms = await qa2(prompt)

        col1, col2 = st.columns(2)
        with col1:
            st.chat_message("assisstant").write(msg1)
        with col2:
            st.chat_message("assisstant").write(msg2)

        # Save retrieved contexts and expanded terms
        st.session_state['retrieved_contexts'] = retrieved_contexts
        st.session_state['expanded_terms'] = expanded_terms

        # Box to display expanded terms
        with st.expander("Expanded Terms"):
            st.write(", ".join(expanded_terms))

        # Box to display retrieved contexts
        with st.expander("Retrieved Contexts"):
            for i, doc in enumerate(retrieved_contexts):
                st.write(f"**Context {i + 1}:** {doc.page_content}")

        st.session_state.question = prompt
        st.session_state.msg1 = msg1
        st.session_state.msg2 = msg2

    if "prompt" in st.session_state:
        stars = st_star_rating("Please rate your experience", maxValue=4, defaultValue=3, key="stars")


async def page_2():
    # Tạo mẫu DataFrame với 4 cột
    df = pd.read_csv('result.csv')

    # Hiển thị DataFrame trong Streamlit
    st.write("Data:")
    st.write(df)
    st.write(f"There is total of {len(df)} answered questons")
    # Visualize cột "score" bằng Matplotlib trong Streamlit
    st.write("Histogram of 'rating' column:")
    fig, ax = plt.subplots()
    ax.hist(df['rating'], bins=4)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')

    # Chỉ hiển thị các giá trị 1, 2, 3, 4 trên trục x
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['1', '2', '3', '4'])

    # Hiển thị giá trị của từng cột trong histogram
    for i, count in enumerate(ax.patches):
        ax.annotate(str(int(count.get_height())),
                    xy=(count.get_x() + count.get_width() / 2, count.get_height()),
                    ha='center', va='bottom')

    st.pyplot(fig)


def page_3():
    # Giao diện nhập API key
    st.title("Nhập API Key")
    api_key_input = st.text_input("API Key:", type="password")
    st.session_state.openai_apikey = api_key_input
    # Nút để lưu API key
    if st.button("Lưu API Key"):
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
        st.session_state.openai_apikey = os.getenv('OPENAI_API_KEY')
    # asyncio.run(question_answering())

    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Select an option", list(PAGES.keys()))
    # Call the page function
    if choice != "Update API KEY":
        asyncio.run(PAGES[choice]())
    else:
        PAGES[choice]()

if __name__ == "__main__":
    main()


