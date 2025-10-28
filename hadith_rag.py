from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone as PineconeBaseClient
from dotenv import load_dotenv
import json
import os

load_dotenv()

class HadithRAG:
    def __init__(self, json_path='hadiths_updated.json'):
        """Initialize RAG system"""
        
        # Load full hadith texts from JSON
        print("📖 Loading hadiths from JSON...")
        with open(json_path, 'r', encoding='utf-8') as f:
            hadiths_list = json.load(f)
            # Create lookup dict by hadith_no
            self.hadiths = {h['hadith_no']: h for h in hadiths_list}
        
        print(f"✅ Loaded {len(self.hadiths)} hadiths")
        
        # Initialize embeddings
        print("🔄 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5',
            model_kwargs={"device": "cuda"}
        )
        
        # Connect to Pinecone
        pc = PineconeBaseClient(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(os.getenv("PINECONE_INDEX_NAME_SQ_V2"))
        
        print("✅ RAG system ready!")
    
    def retrieve(self, query, k=10):
        """Retrieve relevant hadiths with full text"""
        
        # Create query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        # Lookup full texts from JSON
        retrieved_hadiths = []
        for match in results['matches']:
            hadith_no = match['metadata']['hadith_no']
            
            # Get full hadith data from JSON
            if hadith_no in self.hadiths:
                hadith_data = self.hadiths[hadith_no]
                
                retrieved_hadiths.append({
                    'hadith_no': hadith_no,
                    'page_range': match['metadata'].get('page_range', 'N/A'),  # ← Use .get()
                    'section_type': match['metadata'].get('section_type', 'hadith'),  # ← Use .get()
                    'text': hadith_data['hadith_text'],
                    'score': match['score']
                })
        
        return retrieved_hadiths
    
    def format_context(self, retrieved_hadiths):
        """Format hadiths for LLM context"""
        formatted = []
        
        for h in retrieved_hadiths:
            if h['section_type'] == 'preface':
                header = f"[Preface, Pages {h['page_range']}]"
            else:
                header = f"[Hadith No. {h['hadith_no']}, Pages {h['page_range']}]"
            
            formatted.append(f"{header}\n{h['text']}\n---")
        
        return "\n\n".join(formatted)
    
    def query(self, question, k=10):
        """Complete query pipeline"""
        print(f"\n🔎 Query: {question}\n")
        
        # Retrieve relevant hadiths
        retrieved = self.retrieve(question, k=k)
        print(f"📚 Retrieved {len(retrieved)} relevant hadiths")
        
        # Format context
        context = self.format_context(retrieved)
        
        return context, retrieved

# Test usage
if __name__ == "__main__":
    # Initialize RAG
    rag = HadithRAG()
    
    # Test query
    query = "Few people attacked the house of Lady Fatimah (SA) and burned the door"
    context, results = rag.query(query, k=5)
    
    print("\n" + "=" * 70)
    print("📋 RETRIEVED HADITHS:")
    print("=" * 70)
    
    for r in results:
        print(f"\n  Hadith No. {r['hadith_no']} (Pages {r['page_range']})")
        print(f"  Relevance: {r['score']:.4f}")
        print(f"  Preview: {r['text'][:150]}...")
    
    print("\n" + "=" * 70)
    print("📝 FORMATTED CONTEXT (for LLM):")
    print("=" * 70)
    print(context[:1000] + "...")