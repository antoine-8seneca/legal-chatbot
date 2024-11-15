import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI

INDEX_NAME = "llm-chatbot-gpt-new"


def initialize_retriever(user_openai_api_key: str):
    """
    Initializes the embedding and vector store with the user's OpenAI API key.

    Args:
        user_openai_api_key (str): The user's OpenAI API key.

    Returns:
        dict: A dictionary containing the embedding and vector store instances.
    """
    if not user_openai_api_key:
        raise ValueError("User-provided OpenAI API Key is missing. Please provide a valid API key.")
    
    # Initialize OpenAI embedding with user's API key
    embedding = OpenAIEmbeddings(openai_api_key=user_openai_api_key)
    pVS = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding)

    # Return initialized instances
    return {
        "embedding": embedding,
        "pVS": pVS,
        "client": OpenAI(api_key=user_openai_api_key)
    }


def expand_query_with_gpt(query: str, client: OpenAI) -> list:
    """
    Expands a given query using GPT to extract essential legal keywords.

    Args:
        query (str): The user's query.
        client (OpenAI): An instance of the OpenAI client initialized with the user's API key.

    Returns:
        list: A list of expanded keywords or phrases.
    """
    prompt = f"""
    Bạn là một luật sư chuyên nghiệp về lĩnh vực luật hôn nhân và gia đình tại Việt Nam.
    Nhiệm vụ của bạn là trích xuất các cụm từ khóa ngắn gọn, chỉ giữ lại các cụm từ **thực sự cần thiết** liên quan đến câu hỏi, 
    nhằm giúp tìm kiếm thông tin pháp luật chính xác hơn. Không thêm từ khóa thừa.

    Câu hỏi: "{query}"

    Chỉ trả về danh sách các từ khóa hoặc cụm từ liên quan, cách nhau bằng dấu phẩy, ví dụ:
    Ví dụ:
    Câu hỏi: "Tôi có thể kết hôn với cháu ruột của thím mình không?"
    Đầu ra: "kết hôn với cháu ruột, quan hệ huyết thống, luật hôn nhân, hôn nhân huyết thống"

    Dựa trên câu hỏi của người dùng, hãy trích xuất các cụm từ khóa ngắn gọn.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Bạn là một chuyên gia pháp luật."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.5
        )
        expanded_terms = response.choices[0].message.content.strip()
        return expanded_terms.split(", ")
    except Exception as e:
        raise RuntimeError(f"Error in query expansion: {e}")


def retrieve_documents(query: str, k: int, pVS: PineconeVectorStore) -> list:
    """
    Retrieves similar documents from the Pinecone vector store based on a given query.

    Args:
        query (str): The query to search in Pinecone.
        k (int): The number of results to retrieve.
        pVS (PineconeVectorStore): The Pinecone vector store instance.

    Returns:
        list: A list of documents retrieved from the Pinecone vector store.
    """
    try:
        return pVS.similarity_search(query=query, k=k)
    except Exception as e:
        raise RuntimeError(f"Error retrieving documents: {e}")


def process_query(query: str, user_openai_api_key: str, k: int = 3) -> dict:
    """
    Expands the query, retrieves relevant documents, and returns the results.

    Args:
        query (str): The user's query.
        user_openai_api_key (str): The user's OpenAI API key.
        k (int): The number of results to retrieve.

    Returns:
        dict: A dictionary containing expanded keywords and retrieved documents.
    """
    try:
        # Initialize retriever with the user's OpenAI API key
        retriever = initialize_retriever(user_openai_api_key)
        client = retriever["client"]
        pVS = retriever["pVS"]

        # Step 1: Expand the query using GPT
        expanded_terms = expand_query_with_gpt(query, client)
        expanded_query = " ".join(expanded_terms)

        # Step 2: Retrieve documents from Pinecone
        documents = retrieve_documents(expanded_query, k, pVS)

        return {
            "expanded_terms": expanded_terms,
            "documents": documents
        }
    except Exception as e:
        raise RuntimeError(f"Error processing query: {e}")
