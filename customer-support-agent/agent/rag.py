"""
Retrieval-Augmented Generation setup.

Uses Chroma (open-source, runs locally, no server required) as the vector
store and OpenAI embeddings to index the policy documents in
data/knowledge_base/. The index is persisted to disk so it only needs to be
built once.
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")


def _load_and_split_docs():
    loader = DirectoryLoader(KB_DIR, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    return splitter.split_documents(docs)


def get_vectorstore(rebuild: bool = False) -> Chroma:
    """Load the persisted Chroma index, building it from the knowledge base
    on first run (or when rebuild=True)."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    already_built = os.path.isdir(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0

    if rebuild or not already_built:
        splits = _load_and_split_docs()
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name="support_kb",
        )
    else:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="support_kb",
        )
    return vectorstore


def get_retriever(k: int = 3):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
