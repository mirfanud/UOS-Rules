from pathlib import Path
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

documents = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

for pdf_file in Path(".").glob("*.pdf"):

    print(f"Reading {pdf_file.name}")

    try:
        reader = PdfReader(pdf_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

        chunks = splitter.split_text(text)

        for chunk in chunks:

            documents.append(
                {
                    "text": chunk,
                    "source": pdf_file.name
                }
            )

    except Exception as e:

        print(pdf_file.name, e)

texts = [d["text"] for d in documents]

metadatas = [
    {"source": d["source"]}
    for d in documents
]

db = Chroma.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=OpenAIEmbeddings(),
    persist_directory="vector_db"
)

print(f"Indexed {len(texts)} chunks")
