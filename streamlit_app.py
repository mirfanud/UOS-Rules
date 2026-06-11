import streamlit as st
import os


# ----------------------------
# Load environment variables
# ----------------------------

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# ----------------------------
# Streamlit UI setup
# ----------------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(
    page_title="UOS Rules Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University Rules Assistant (UOS)")
st.write("Ask questions about rules, policies, acts, and regulations.")

# ----------------------------
# Load Vector DB (Chroma)
# ----------------------------
@st.cache_resource
def load_db():
    embeddings = OpenAIEmbeddings(
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    db = Chroma(
        persist_directory="vector_db",
        embedding_function=embeddings
    )

    return db

db = load_db()

# ----------------------------
# Load LLM
# ----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",   # you can upgrade later
    api_key=st.secrets["OPENAI_API_KEY"]
)

# ----------------------------
# Retriever function
# ----------------------------
def ask_question(question):

    docs = db.similarity_search(question, k=5)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a strict university regulations assistant.

Answer ONLY using the provided context.

If the answer is not in the context, say:
"I could not find this information in the provided university documents."

Context:
{context}

Question:
{question}

Provide a clear and structured answer.
"""

    response = llm.invoke(prompt)

    return response.content, docs

# ----------------------------
# Chat input
# ----------------------------
question = st.chat_input("Ask your question about university rules...")

if question:

    with st.spinner("Searching university regulations..."):

        answer, docs = ask_question(question)

    # ----------------------------
    # Answer section
    # ----------------------------
    st.subheader("📌 Answer")
    st.write(answer)

    # ----------------------------
    # Sources section
    # ----------------------------
    st.subheader("📚 Sources")

    shown = set()

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")

        if source not in shown:
            shown.add(source)

            with st.expander(source):
                st.write(doc.page_content[:1500])
