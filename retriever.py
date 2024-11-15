import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "llm-chatbot-gpt-new"

# Initialize embedding and Pinecone vector store
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pVS = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding)


def expand_query_with_gpt(query: str) -> list:
    """
    Expands a given query using GPT to extract essential legal keywords.

    Args:
        query (str): The user's query.

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


def retrieve_documents(query: str, k: int = 3) -> list:
    """
    Retrieves similar documents from the Pinecone vector store based on a given query.

    Args:
        query (str): The query to search in Pinecone.
        k (int): The number of results to retrieve.

    Returns:
        list: A list of documents retrieved from the Pinecone vector store.
    """
    try:
        return pVS.similarity_search(query=query, k=k)
    except Exception as e:
        raise RuntimeError(f"Error retrieving documents: {e}")


def process_query(query: str, k: int = 3) -> dict:
    """
    Expands the query, retrieves relevant documents, and returns the results.

    Args:
        query (str): The user's query.
        k (int): The number of results to retrieve.

    Returns:
        dict: A dictionary containing expanded keywords and retrieved documents.
    """
    try:
        # Step 1: Expand the query using GPT
        expanded_terms = expand_query_with_gpt(query)
        expanded_query = " ".join(expanded_terms)

        # Step 2: Retrieve documents from Pinecone
        documents = retrieve_documents(expanded_query, k)

        return {
            "expanded_terms": expanded_terms,
            "documents": documents
        }
    except Exception as e:
        raise RuntimeError(f"Error processing query: {e}")
