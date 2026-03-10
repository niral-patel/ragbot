from langchain_community.document_loaders import PyPDFLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
import os

def load_chunk(pdf_path: str):
    """_summary_
    Load a PDF file and split it into chunks.
    Args:
        pdf_path (str): _description_
    """