#from langchain_core.text_splitter import CharacterTextSplitter;
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader



import os
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent.parent / "data/documents/md"
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]   # paragraph > line > sentence > word > char

def load_document():
    docs = []
    for dir in DATA_DIR.iterdir():
        if not dir.is_dir() and dir.is_file() and dir.suffix == ".md":
            file = dir
            print(f"Processing file: {file}")
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={"source": file.name}))

    return docs

def perform_recursive_chunking(document, chunk_size=30, chunk_overlap=5):
    """
    Perform recursive chunking on a document.

    Args:
        document (Document): The input document to be chunked.
        chunk_size (int): The size of each chunk in characters.
        chunk_overlap (int): The number of overlapping characters between chunks.   
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(document.page_content)

    print(f"Document: {document.metadata['source']}")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk}...")  # Print the chunk

def main():
    documents = load_document()
    for doc in documents:
        print(f"Document Name: {doc.metadata['source']}...")  
        perform_recursive_chunking(doc, chunk_size=20, chunk_overlap=10)


if __name__ == "__main__":
    main()