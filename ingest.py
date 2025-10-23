from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time
import os

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "hayat-qulub-alama-majlisi"
EMBEDDINGS_DIM = 384

# ---------- begin of pipelines ----------
# load pdf
pdf_path = 'Hayat-Qulub-Alama-Majlisi.pdf'
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = splitter.split_documents(docs)

# create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L12-v2", kwargs={"device": "cpu"}
)

# Store in vector database - FAISS
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local("Hayat-Qulub-Alama-Majlisi-faiss-index")

# store in vector database - Pinecone
index_name = "hayat-qulub-alama-majlisi"
if pc.has_index(index_name):
    index = pc.create_index_from_model(
        name=index_name,
        region="us-east-1",
        embed = {
            "model": "all-MiniLM-L12-v2",
            "dimension": 384
        }
    )
    time.sleep(5)

pinecone_store = Pinecone.from_documents(
    documents=chunks,
    embedding=EMBEDDINGS_DIM,
    index_name=INDEX_NAME,
    metric="cosine",
)

# ---------- end of pipelines ----------
 