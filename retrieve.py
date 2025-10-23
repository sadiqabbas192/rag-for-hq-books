from pinecone import Pincone
from langchain_google_genai import ChatGoogleGenerativeAI
from ingest import pinecone_store
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.retrievers import RetrievalQA
import os

load_dotenv()
API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
API_KEYS = [key.strip() for key in API_KEYS if key.strip()]
current_key_index = 0

if not API_KEYS:
    raise ValueError("No API keys found. Please set the GEMINI_API_KEY environment variable.")

# get top-k values
retrieve = pinecone_store.retriever(search_type="similarity", search_kwargs={"k": 10})

# Prompt template
system_prompt = """
You are an Islamic history assistant. 
Always answer in a respectful and storytelling way. 
If the answer is not in the documents, say "I don’t know based on my knowledge."
Question: {question}
Context: {context}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=system_prompt
)

# LLM's
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key = API_KEYS[current_key_index],
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retrieve,
    chain_type_kwargs={"prompt": prompt}
)

def ask (question: str):
    global current_key_index
    try:
        return qa.run(question)
    except Exception as e:
        print(f"Error occurred: {e}")
        current_key_index = (current_key_index + 1) % len(API_KEYS)
        llm.api_key = API_KEYS[current_key_index]
        print(f"Switched to API key index: {current_key_index}")
        return qa.run(question)
