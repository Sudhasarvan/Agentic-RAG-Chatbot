import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Tuple

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


from openai import OpenAI
import requests

# ------------------ CONFIG ------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

SIMILARITY_THRESHOLD = 1.5
TOP_K = 4

# ------------------ PDF LOADING ------------------
def load_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# ------------------ CHUNKING ------------------
def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200

    )
    chunks = splitter.split_text(text)
    print(f"✅ Number of chunks created: {len(chunks)}")
    return chunks


# ------------------ VECTOR STORE ------------------
@st.cache_resource
def create_vector_store(chunks: List[str]):
    docs = [Document(page_content=c) for c in chunks]
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# ------------------ RETRIEVAL ------------------
def retrieve_docs(query: str, vectorstore) -> Tuple[List[Document], float]:
    docs = vectorstore.similarity_search_with_score(query, k=TOP_K)

    retrieved_docs = [d[0] for d in docs]
    scores = [d[1] for d in docs]

    best_score = min(scores) if scores else 999


    print(f"📊 Similarity score: {best_score}")
    print("Scores:", scores)

    return retrieved_docs, best_score



# ------------------ RELEVANCE CHECK ------------------
def is_relevant(score: float) -> bool:
    return score < SIMILARITY_THRESHOLD


# ------------------ WEB SEARCH ------------------
def search_web(query: str) -> str:
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"q": query}

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    snippets = []
    if "organic" in data:
        for item in data["organic"][:5]:
            snippets.append(item.get("snippet", ""))

    return "\n".join(snippets)


# ------------------ LLM: PDF ANSWER ------------------
def generate_answer_pdf(query: str, docs: List[Document]) -> str:
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful assistant.

Answer the question ONLY using the provided context.

If the answer is partially available, try to infer carefully.
Only say "NOT FOUND" if absolutely no relevant information exists.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


# ------------------ LLM: WEB ANSWER ------------------
def generate_answer_web(query: str, web_data: str) -> str:
    prompt = prompt = f"""
You are a knowledgeable assistant.

Use the web data below to answer the question clearly.

Always provide a useful answer even if data is incomplete.
Do NOT say "I cannot answer".

Web Data:
{web_data}

Question:
{query}

Provide a clear answer and include sources.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


# ------------------ ROUTER (AGENTIC LOGIC) ------------------
def router(query: str, vectorstore):
    docs, score = retrieve_docs(query, vectorstore)

    if is_relevant(score):
        print("✅ Routing: PDF")
        answer = generate_answer_pdf(query, docs)

        if "NOT FOUND" in answer:
            print("⚠️ Fallback triggered → Web")
            web_data = search_web(query)
            answer = generate_answer_web(query, web_data)
            return answer, "web"

        return answer, "pdf"

    else:
        print("🌐 Routing: Web (low relevance)")
        web_data = search_web(query)
        answer = generate_answer_web(query, web_data)
        return answer, "web"


# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")

st.title("🤖 Agentic RAG Chatbot (PDF + Web Fallback)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

vectorstore = None

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = load_pdf(uploaded_file)
        chunks = chunk_text(text)
        vectorstore = create_vector_store(chunks)

    st.success("PDF processed successfully!")

query = st.text_input("Ask a question")

if query and vectorstore:
    with st.spinner("Thinking..."):
        answer, source = router(query, vectorstore)

    st.session_state.chat_history.append((query, answer, source))

# ------------------ CHAT DISPLAY ------------------
for q, a, s in st.session_state.chat_history:
    st.markdown(f"**🧑 You:** {q}")
    if s == "pdf":
        st.markdown(f"📄 **From PDF:** {a}")
    else:
        st.markdown(f"🌐 **From Web:** {a}")
