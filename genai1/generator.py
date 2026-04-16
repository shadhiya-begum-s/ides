#generator.py 
import os
import textwrap
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
FALLBACK_MAX_TOKENS = int(os.getenv("FALLBACK_MAX_TOKENS", 500))

_client = None

def get_groq_client():
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set.")
        _client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE)
    return _client

def build_doc_context(retrieved_chunks):
    pieces = []
    for i, (meta, score, idx) in enumerate(retrieved_chunks, start=1):
        src = meta.get("source", "unknown")
        page = meta.get("page")
        chunk_index = meta.get("chunk_index", "?")
        snippet = meta.get("text", "").strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + " ...[truncated]"
        if page:
            ref_label = f"{src}: Page {page}"
        else:
            ref_label = f"{src}: Paragraph {chunk_index}"
        pieces.append(f"[{i}] {ref_label}\n{snippet}")
    return "\n\n---\n\n".join(pieces)
def generate_answer(question: str, retrieved_chunks: list, temperature: float = 0.0, max_tokens: int = 500):
    client = get_groq_client()
    context = build_doc_context(retrieved_chunks) if retrieved_chunks else ""

    if context:
        user_msg = textwrap.dedent(f"""  
        You are an assistant that MUST answer using the provided document context below.  
        If the context does NOT contain the answer, use your general knowledge to answer.  
        Provide citations only for the document context.  

        Context:  
        {context}  

        Question:  
        {question}  

        Rules:  
        - Answer concisely.  
        - Provide explicit citations for any document content in format: (Source: <document name> — Page <n>)  
        - If using general knowledge because documents lack info, append "(Generated — not in documents)".  
        """).strip()
    else:
        user_msg = textwrap.dedent(f"""  
        Answer the following question using general knowledge.  
        Question:  
        {question}  
        """).strip()

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers using provided documents or general knowledge."},
            {"role": "user", "content": user_msg}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    answer = resp.choices[0].message.content.strip()
    is_generated = "(Generated — not in documents)" in answer

    return answer, is_generated
#cd your-project-folder
#python -m venv .venv
#.venv\Scripts\activate
#pip install -r requirements.txt
#streamlit run app.py