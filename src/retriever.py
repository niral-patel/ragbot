#retriever.py: This file will contain the logic for retrieving relevant chunks from the vector database based on a user query.
# The Retriever class will take a user query, convert it into an embedding, and then search the vector database for the most relevant chunks.
# This is a crucial part of the RAG pipeline, 
# as it ensures that the LLM receives the most relevant information to generate accurate and contextually appropriate responses.
# PURPOSE  : Embed chunks into vectors and store/search in ChromaDB
# USES     : logger.py, exception.py, ingestor.py
# CALLED BY: src/chain.py (next file)

import os
import sys
import time

from src.logger import logger
from src.exception import CustomException

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from chromadb import Client

#-- Constant for ChromaDB collection name ---
CHROMA_COLLECTION_NAME = "ragbot_collection"
CHROMA_PATH = os.path.join(os.getcwd(), 'chroma_db')  # Store ChromaDB files in project folder
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace embedding model (small & fast)

# WHY this embedding model?
# - Free, runs locally on your CPU (no API cost)
# - Downloads once (~90MB), cached forever
# - Converts text to 384-dimensional vectors
# - Fast enough for real-time search
# - Used in production by many companies

class VectorRetriver:
    '''
    Handles embedding chunks, storing and semantic searching in ChromaDB.

    Two main responsibilities:
    1. store()  — embed chunks and save to ChromaDB (done ONCE per document)
    2. search() — find most relevant chunks for a query (done on every question)
    
    HOW TO USE:
        from src.retriever import VectorRetriever

        retriever = VectorRetriever()
        retriever.create_collection()  # Create ChromaDB collection (only once)
        retriever.add_chunks(chunks)   # Add list of Document chunks to the collection
        results = retriever.search(query, top_k=5)  # Search for relevant chunks based on user query
    This class is responsible for embedding the chunks into vectors and storing them in ChromaDB.
    It also provides a search method that takes a user query, converts it into an embedding, and retrieves the most relevant chunks from the collection based on cosine similarity.
    '''

    def __init__(self):
        try:
            logger.info("VectorRetriever initialized")
            # Load embedding model (HuggingFace) once during initialization
            self.embedding_function = self._load_embedding_model()
            logger.info("VectorRetriever initialized successfully.")
            
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def _load_embedding_model(self) -> HuggingFaceEmbeddings:
        '''
        Load the HuggingFace embedding model. This is done once during initialization to avoid reloading the model every time we need to embed text.
        Downloads on first run (~90MB), cached locally after that.
        Returns:
            An instance of HuggingFaceEmbeddings that can be used to convert text to vectors.

        Raises:
            CustomException: If there is an error loading the embedding model.
            
        WHY CPU and not GPU?
        Embedding is fast on CPU. Saving GPU memory for the LLM (Phi-3)
        which needs it more. This is a common production pattern.
        '''
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            logger.info('(Downloading model for the first time? This may take a minute... ~90MB on first run, cached locally after that)')
            start = time.time()
            embedding_function = HuggingFaceEmbeddings(
                model_name   = EMBEDDING_MODEL,
                model_kwargs = {"device": "cpu"},  # Force CPU usage for embedding
                encode_kwargs= {"normalize_embeddings": True}  # Normalize vectors for better cosine similarity search, vectors scaled to length 1
                #This makes cosine similarity more accurate and is a common practice in production vector search systems
            )
            duration = round(time.time() - start, 2)
            logger.info(f"Embedding model loaded successfully in {duration} seconds.")
            
            return embedding_function
        
        except Exception as e:
            raise CustomException(e, sys)

    def store(self, chunks: list) -> Chroma:
        '''
        Embed all the chunks and store them in ChromaDB. This should be called once after chunking a document.

        WHEN TO CALL THIS:
        - Only when you add a NEW document
        - Not on every question — that would be very slow

        HOW IT WORKS:
        1. Takes your 23 chunks
        2. Passes each through embedding model → 23 vectors of 384 numbers
        3. Stores vectors + original text + metadata in ChromaDB folder

        Args:
            chunks : list of Document objects from DataIngestion.run()

        Returns:
            Chroma vectorstore object that can be used for searching later.
        
        Args:
            chunks: List of Document objects (output from ingestor) that need to be embedded and stored.
        Raises:
            CustomException: If there is an error during embedding or storing in ChromaDB.
        '''
        try:
            #-- Sanity check: Ensure we have chunks to store ---
            # ── Clear existing ChromaDB to prevent duplicates ──────────────
            # In production you would check document IDs instead
            # For development: always start fresh when re-ingesting
            import shutil
            if os.path.exists(CHROMA_PATH):
                shutil.rmtree(CHROMA_PATH)
                logger.info(f"Cleared existing ChromaDB at: {CHROMA_PATH}/")

            # ... rest of your code stays exactly the same
                
                
            logger.info(f"Embedding and storing {len(chunks)} chunks in ChromaDB...")
            logger.info("This may take a moment, especially on the first run when the embedding model is downloaded.")
            logger.info(f"saving to: {CHROMA_PATH}/ (ChromaDB files will be stored here)")
            
            start = time.time()
            
            db = Chroma.from_documents(
                documents = chunks,  # Original text of each chunk and Metadata (e.g., source document, chunk index
                embedding = self.embedding_function,  # Use the loaded embedding function to convert text to vectors
                collection_name = CHROMA_COLLECTION_NAME,  # Name of the ChromaDB collection to store the chunks
                persist_directory = CHROMA_PATH  # Directory where ChromaDB files will be stored
                # ChromaDB will handle creating the collection and storing the vectors + metadata in the specified directory.
                # persist_directory → saves to disk
                # Without this, database disappears when Python exits
            )
            
            duration = round(time.time() - start, 2)
            logger.info(
                f" {len(chunks)} Chunks embedded and stored in ChromaDB successfully in {duration} seconds."
                )
            
            return db
            
        except Exception as e:
            raise CustomException(e, sys)
            
    def load(self) -> Chroma:
        '''
        Load the existing ChromaDB collection from disk. This should be called before searching if you have already stored chunks.

        WHEN TO CALL THIS:
        - Every time user asks a question
        - Much faster than re-embedding (just loads from disk)
        - Before calling search() if you have previously stored chunks
        - Not needed if you just called store() since it returns the db object

        HOW IT WORKS:
        1. Looks for ChromaDB files in the specified directory
        2. Loads the collection into memory so you can perform searches on it

        Returns:
            Chroma vectorstore object that can be used for searching.
        Raises:
            CustomException: If there is an error loading the ChromaDB collection (e.g., collection not found).
        '''
        try:
            logger.info(f'Loading existing ChromaDB collection from {CHROMA_PATH}/...')
            
            if not os.path.exists(CHROMA_PATH):
                raise CustomException(
                    f"No ChromaDB collection found at {CHROMA_PATH}/."
                    " Please run store() to create the collection first.", sys)
            
            db = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                persist_directory=CHROMA_PATH,
                embedding_function=self.embedding_function
            )
            
            logger.info("ChromaDB collection loaded successfully.")
            return db
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def search(self, query: str, top_k: int = 5) -> list:
        '''
        Search for the most relevant chunks in ChromaDB based on a user query.
        Find the k most relevant chunks for a given query.
        HOW TO USE:
        1. Call load() to load the existing ChromaDB collection (if not already loaded)
        2. Call search(query, top_k) with the user's question to retrieve relevant chunks

        HOW IT WORKS:
        1. Takes user query (e.g., "What is RAG?")
        2. Converts query to embedding vector using the same embedding model
        3. Searches ChromaDB for top_k most similar vectors (chunks) based on cosine similarity
        4. Returns the original text of those chunks along with metadata
        
        HOW SIMILARITY SEARCH WORKS:
        1. Convert query text → vector (384 numbers)
        2. Compare to all stored vectors using cosine similarity
        3. Return top-k closest matches

        Cosine similarity score:
          1.0  = identical meaning
          0.0  = completely unrelated
          Typical good match = 0.3 to 0.7 (depends on the use case)
        
        Args:
            query: The user's question or query string.
            top_k: The number of most relevant chunks to retrieve (default is 5).

        Returns:
            A list of dictionaries containing the original text and metadata of the most relevant chunks. List of (Document, score) tuples where Document contains page_content and metadata, and score is the similarity score.

        Raises:
            CustomException: If there is an error during the search process (e.g., collection not found, search failure).
        '''
        try:
            logger.info(f"Searching for relevant chunks for query: '{query}' with top_k={top_k}...")
            
            # Load existing ChromaDB collection
            db = self.load()
            
            # Perform semantic search in ChromaDB
            results = db.similarity_search_with_score(query, k=top_k)
            
            logger.info(f"Search completed successfully. Found {len(results)} relevant chunks.")
             # Log each result so you can see what was found and the similarity score
            for i, (doc, score) in enumerate(results):
                logger.info(
                    f"  Result {i+1}: score={score:.4f} | "
                    f"page={doc.metadata.get('page','?')} | "
                    f"preview={doc.page_content[:60].strip()}..."
                )
            
            return results
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
#-- Run Directly to test embedding and storing chunks in ChromaDB ---
if __name__ == "__main__":
    
    from src.ingestor import DataIngestion
    
    # Sample document for testing
    pdf_path = "data/DailyReport.pdf" # Repeat to increase length
    
     # ── Check command line argument ────────────────────────────────────────
    # python src/retriever.py store   → ingests and stores
    # python src/retriever.py search  → searches only (no re-ingestion)
    # python src/retriever.py         → does both (default)
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    retriever = VectorRetriver()
    try:
        
        if mode in ("store", "both"):
            # Step 1: Ingest and chunk the document
            ingestor = DataIngestion()
            chunks = ingestor.run(pdf_path)
            # Step 2: Create VectorRetriever and store chunks in ChromaDB
            retriever.store(chunks)
            
        if mode in ("search", "both"):        
            # Step 3: Search
            print("\n" + "="*50)
            print("Testing semantic search...")
            print("="*50)

            query   = "What is the total room revenue?"
            results = retriever.search(query, top_k=3)

            print(f"\nQuery  : {query}")
            print(f"Results: {len(results)} chunks found\n")

            for i, (doc, score) in enumerate(results):
                print(f"[{i+1}] Score : {score:.4f}")
                print(f"     Page  : {doc.metadata.get('page', '?')}")
                print(f"     Text  : {doc.page_content[:150].strip()}")
                print()

        
    except Exception as e:
        logger.error(f"Error during testing: {e}")