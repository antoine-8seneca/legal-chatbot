# split text

from langchain_core.documents.base import Document
from pypdf import PdfReader
import os
import time

path = 'data/pdf/'
docs = []

for file in os.listdir(path):
    reader = PdfReader(f"{path}{file}")
    print(f"{path}{file}")
    # read content of file
    content = ""
    for page in reader.pages:
        content += page.extract_text()

    # convert to Document type
    chunks = content.split('-')
    for chunk in chunks:
        if (len(chunk) != 0):
            doc = Document(
                page_content=chunk,
                metadata={"source":f"{path}{file}"}
            )

            docs.append(doc)
    print(len(docs))

# embedding and upsert to vectorstore

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
    name="llm-chatbot-gpt-new",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",
    ),

)

from langchain_community.embeddings.openai import OpenAIEmbeddings

# embedding technique of OpenAI
embedding=OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
embedding

pVS = PineconeVectorStore(
    index_name="llm-chatbot-gpt-new",
    embedding=embedding
)

pVS.add_documents(
    documents=docs[:200],
)

time.sleep(10)
pVS.add_documents(
    documents=docs[201:400],
)

time.sleep(10)
pVS.add_documents(
    documents=docs[401:],
)


