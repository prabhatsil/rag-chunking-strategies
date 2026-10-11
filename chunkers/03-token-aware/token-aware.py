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
from transformers import AutoTokenizer
import tiktoken
from langchain_community.document_loaders import PyPDFLoader



import os
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent.parent / "data/documents/md"
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]   # paragraph > line > sentence > word > char

def load_document():
    docs = []
    for dir in DATA_DIR.iterdir():
        if not dir.is_dir() and dir.is_file() and dir.name == "test_token_aware.md":
            file = dir
            
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                print(f"Processing content : {len(content)} characters from {file}")
            docs.append(Document(page_content=content, metadata={"source": file.name}))

    return docs

def perform_huggingface_token_chunking(document, tokenizer_name, chunk_size=30, chunk_overlap=5):
    """
    Perform recursive chunking on a document.

    Args:
        document (Document): The input document to be chunked.
        chunk_size (int): The size of each chunk in characters.
        chunk_overlap (int): The number of overlapping characters between chunks.   
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
     
    chunks = token_splitter.split_text(document.page_content)

    print(f"Document: {document.metadata['source']}")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        token_count =  len(tokenizer.encode(chunk, add_special_tokens=False)) # Tokenize the chunk
        print(f"Chunk {i + 1}: Token Count: {token_count} Chunk : {chunk}...")  # Print the chunk
    
def perform_token_aware_chuking(document, chunk_size=30, chunk_overlap=5):
        token_splitter =  RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                            encoding_name="cl100k_base", 
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap)
        chunks = token_splitter.split_text(document.page_content)
        
        print(f"Number of chunks using tiktokenizer : {len(chunks)}")
        encoding = tiktoken.get_encoding("cl100k_base")
        for i, chunk in enumerate(chunks):
            token_count =  len(encoding.encode(chunk)) # Tokenize the chunk
            print(f"Chunk {i + 1}: Token Count: {token_count} Chunk : {chunk}...")  # Print the chunk
    

def main():
    documents = load_document()
    print(f"Total documents loaded: {len(documents)}")
    tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"
    for doc in documents:
        print(f"Document Name: {doc.metadata['source']}...")  
          # Specify the tokenizer name
        perform_token_aware_chuking(doc,  chunk_size=50, chunk_overlap=8)


if __name__ == "__main__":
    main()