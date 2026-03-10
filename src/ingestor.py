# ingester.py : Load the PDF file and split it into chunks.
#LLM can't read the whole PDF file, so we need to break it into bite-sized pieces that can be easily searched and retrieved.

#importing necessary libraries
import os
import logging
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_chunk(pdf_path: str):
    
    """_summary_
    Step - 1:
    Load a PDF file and split it into chunks using PyPDFloader and RecursiveCharacterTextSplitter from the langchain_community library.
    each page becomes a Documet object, and then we split the text into smaller chunks of 1000 characters with an overlap of 200 characters to ensure that we don't lose any context between chunks.
    .page_content = the text content of the page, and .metadata = the metadata of the page (e.g., page number, file name, etc.)
    .metadata - a dictionary that contains metadata about the document, such as the page number, file name, etc. This metadata can be useful for later retrieval and reference. {"source": "example.pdf", "page": 0}
    Args:
        pdf_path (str): _description_
    """
    
    # Load the PDF file
    print(f'Loading PDF file: {pdf_path}')
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f' -> Loaded {len(documents)} pages from the PDF file.')

    #Step - 2: Split the text into chunks.
    # RecursiveCharacterTextSplitter is a text splitter that splits text into chunks of a specified size with a specified overlap. It is called "recursive" because it can split text recursively until the desired chunk size is reached. This is useful for splitting long documents into smaller pieces that can be easily processed and analyzed.
    # RecursiveCharacterTextSplitter tries to split on:
    #   paragraphs first → then sentences → then words → then characters
    # This keeps meaning intact as much as possible
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        )
    chunks = text_splitter.split_documents(documents)
    print(f' -> Split into {len(chunks)} chunks.')
    
    #Check the chunks 
    for i, chunk in enumerate(chunks[:3]):
        print('--------------Chunk Preview--------------\n')
        print(f'Chunk {i+1}:')
        print(f'Content: {chunk.page_content[:200]}...')  # Print the first 200 characters of the chunk
        print(f'Metadata: {chunk.metadata}')
        print('---------------------------------\n')

    return chunks

if __name__ == "__main__":
    # Example usage
    pdf_path = 'data/DailyReport.pdf'  # Replace with your PDF file path
    
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' does not exist. Please check the file path and try again.")
    else:
        chunks = load_chunk(pdf_path)
        print(f"Total chunks ready for embedding: {len(chunks)}")