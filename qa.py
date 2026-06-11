from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

llm = ChatOpenAI(
    model="gpt-5"
)

db = Chroma(
    persist_directory="vector_db",
    embedding_function=OpenAIEmbeddings()
)

def ask_question(question):

    docs = db.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
You are the University of Swabi Regulations Assistant.

Answer ONLY from the provided regulations.

If the answer cannot be found, say:
"I could not locate a rule covering this question."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content, docs
