# ingester.py : Load the PDF file and split it into chunks.
#LLM can't read the whole PDF file, so we need to break it into bite-sized pieces that can be easily searched and retrieved.

#importing necessary libraries
import os
import sys
import time


from src.logger import logger
from src.exception import CustomException

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataIngestion:
    '''
     Handles loading and chunking of PDF documents.

    WHY A CLASS?
    - Keeps all ingestion logic in one place
    - Easy to extend later (add Word docs, URLs, etc.)
    - Can store config (chunk_size, overlap) as attributes
    - Interviewers expect OOP in production code
        HOW TO USE:
        from src.ingestor import DataIngestion
    
        ingestor = DataIngestion()
        chunks = ingestor.load_chunk('path/to/your.pdf')
    
        Each chunk is a Document object with .page_content and .metadata
    This class is responsible for loading the PDF file and splitting it into chunks. 
    It uses the PyPDFLoader to load the PDF file and the RecursiveCharacterTextSplitter to split the text into chunks.
    The load_chunk method takes the path to the PDF file as input and returns a list of chunks, where each chunk is a Document object that contains the text content and metadata of the chunk.
    The metadata includes information about the source of the chunk (e.g., file name, page number) which can be useful for later retrieval and reference.
    '''
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """
        chunk_size    : max characters per chunk (500 = ~100 words)
        chunk_overlap : shared characters between chunks (preserves context)
        """
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(
            f"DataIngestion initialized --"
            f"chunk_size={chunk_size} and chunk_overlap={chunk_overlap}"
        )
        
    def load_pdf(self, pdf_path: str) -> list:
        """
        Load a PDF file and return a list of Document objects, where each Document represents a page in the PDF.

        Args:
            pdf_path : path to PDF file e.g. 'data/test.pdf'

        Returns:
            List of Document objects (one per page)

        Raises:
            CustomException : if file missing, empty, or unreadable
        """
        try:
            #-- Guard: check if file exists and is not empty ---
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(
                    f"PDF not found at '{pdf_path}'. "
                    f"Place your PDF inside the data/ folder."
                )
            
            logger.info(f"Loading PDF file: {pdf_path}")
            start = time.time()

            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            duration = round(time.time() - start, 2)
            logger.info(f"Loaded {len(pages)} pages from the PDF file in {duration} seconds.")
            
            #-- Guard: check if PDF is empty ---
            if len(pages) == 0:
                raise ValueError(
                    f"PDF file '{pdf_path}' is empty."
                    f"Please provide a valid PDF file."
                    )
            
            return pages
        
        except Exception as e:
            #logger.error(f"Error loading PDF file: {pdf_path}")
            raise CustomException(e, sys)
        
        
    def split_into_chunks(self, pages : list) -> list:
        """
        Split a list of pages objects into smaller chunks using RecursiveCharacterTextSplitter (overlapping text chunks).

        Args:
            pages  : List of pages objects (e.g., pages from PDF) load_pdf()

        Returns:
            List of smaller chunked pages objects with .page_content and .metadata

        Raises:
            CustomException : if splitting fails for any reason or produces no chunks
        """
        try:
            logger.info(
                f"Splitting {len(pages)} pages  into chunks."
                f"Chunk size: {self.chunk_size} chars, "
                f"Chunk overlap: {self.chunk_overlap} chars."
                )
            start = time.time()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
            
            chunks = text_splitter.split_documents(pages)
            logger.info(
                f"Created {len(chunks)} chunks"
                )
            
            # -- Guard: check if splitting produced any chunks ---
            if not chunks:
                raise ValueError("No chunks produced after splitting.")
            
            #-- guard must produce chunks with content and metadata ---
            if len(chunks) == 0:
                raise ValueError(
                    "Chunking produced 0 chunks. "
                    "PDF may contain only images with no text."
                )

            # Log first chunk as a sanity check
            logger.info(
                f"First chunk preview: "
                f"{chunks[0].page_content[:100].strip()}..."
            )

            #-- Optional: log the first 3 chunks for debugging ---
            duration = round(time.time() - start, 2)
            logger.info(f"Split into {len(chunks)} chunks in {duration} seconds.")
            
            return chunks
        
        except Exception as e:
            #logger.error("Error splitting documents into chunks.")
            raise CustomException(e, sys)
        
    def run(self, pdf_path: str) -> list:
        """
        Main method to run the data ingestion process: load PDF and split into chunks.
        ingestion pipeline: Calls load_pdf() then split_into_chunks() in sequence.
        Args:
            pdf_path : path to PDF file e.g. 'data/test.pdf'

        Returns:
            List of chunked Document objects ready for embedding

        Raises:
            CustomException : if any step in the process fails
            
        Usage:
            ingestor = DataIngestion()
            chunks   = ingestor.run("data/test.pdf")
        """
        try:
            logger.info("=" * 50)
            logger.info("Starting Data Ingestion Pipeline")
            logger.info("=" * 50)
            
            
            pages = self.load_pdf(pdf_path)
            chunks = self.split_into_chunks(pages)
            
            logger.info("Data Ingestion Complete")
            logger.info(f"Total chunks ready for embedding: {len(chunks)}")
            logger.info("=" * 50)
            return chunks
        
        except Exception as e:
            #logger.error("Data ingestion process failed.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example usage
    pdf_path = 'data/DailyReport.pdf'  # Replace with your PDF file path
    
    try:
         # Create ingestor with default settings
        ingestor = DataIngestion()
        
        # Run full pipeline
        chunks = ingestor.run(pdf_path)
    
         # Summary
        print(f"\n{'='*50}")
        print(f"  Chunks created  : {len(chunks)}")
        print(f"  First chunk     : {chunks[0].page_content[:80]}...")
        print(f"  Metadata        : {chunks[0].metadata}")
        print(f"  Log saved to    : logs/")
        print(f"{'='*50}\n")

    except CustomException as e:
        print(f"\n  ERROR: {e}\n")

    except Exception as e:
        print(f"\n  UNEXPECTED ERROR: {e}\n")