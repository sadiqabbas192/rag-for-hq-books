import os 
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeBaseClient
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from system_prompt import system_prompt_text
load_dotenv()

# Setup Pinecone
pc = PineconeBaseClient(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME_SQ_V2")
index = pc.Index(INDEX_NAME)

# create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": "cpu"}
)

pinecone_store = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME, 
    embedding=embeddings)

# Create retriever from Pinecone
retrieve = pinecone_store.as_retriever(
    search_type='similarity', 
    search_kwargs={"k": 10}
)

# System Prompt
system_prompt = system_prompt_text

prompt = ChatPromptTemplate.from_template(system_prompt)

# LLM and QA
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-exp',
    temperature=0,
    api_key=os.getenv('GEMINI_API_KEY')
)

def format_docs(docs):
    blocks = []
    for d in docs:
        hadith_no = d.metadata.get("hadith_no", "UNKNOWN")
        page_start = d.metadata.get("page_start")
        # ensure hadith_no is integer-like for neatness
        try:
            hadith_tag = f"[HADITH_NO:{int(hadith_no)}]"
        except:
            hadith_tag = f"[HADITH_NO:{hadith_no}]"
        # clean page markers inside content (optional)
        content = d.page_content.replace("(Page", "").replace("Page", "")
        content = content.strip()
        blocks.append(f"{hadith_tag} | PAGE:{page_start}\n{content}")
    return "\n\n".join(blocks)


input = {"context": retrieve | format_docs, 
      "question": RunnablePassthrough()
      }

qa_chain = ( input| prompt | llm | StrOutputParser())

def ask(question: str) -> str:
    """Ask a question and get an answer from the RAG system."""
    result = qa_chain.invoke(question)
    return result