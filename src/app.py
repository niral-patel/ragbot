# app.py : Streamlit web UI for the RAG Chatbot
# This file creates a simple web interface using Streamlit where users can upload PDF files, ask questions, and receive answers based on the content of the uploaded documents. 
# It integrates the DataIngestion and Retriever classes to handle document processing and question answering.
#RUN WITH : streamlit run src/app.py
# USES     : chain.py (which uses retriever.py + ingestor.py)


import os
import sys
import time

import streamlit as st

from src.logger import logger
from src.exception import CustomException
from src.ingestor import DataIngestion
from src.retriever import VectorRetriever
from src.chain import RAGChain

#-- Page configuration for Streamlit
# Set the page title, icon, and layout for the Streamlit app. 
# This helps in creating a more user-friendly interface.
#Must be the first Strimlet command in the script to avoid errors about "set_page_config must be called before any other Streamlit command"

st.set_page_config(
    page_title  ="RAG Document Chatbot",
    page_icon   ="🤖",
    layout      ="wide"
)

# -- Load the RAG Chain components once at startup ----------------
# Initialize the DataIngestion, VectorRetriever, and RAGChain classes.
# This ensures that the necessary components are ready to handle user interactions without delay.
# By loading these components at the start, we avoid re-initializing them on every user action, which improves performance.
# st.cache_resource = load ONCE, reuse forever across all user interactions
# Without this: Phi-3 reloads on EVERY question = 5-10s delay every time
# With this   : Phi-3 loads once at startup = fast responses after the initial load


@st.cache_resource
def load_chain():
    """Load RAGChain once and cache it for the session."""
    try:
        logger.info("Loading RAGChain components...")
        ingestor = DataIngestion()
        retriever = VectorRetriever()
        chain = RAGChain(model_name="Phi3")
        logger.info("RAGChain components loaded successfully.")
        return chain
    
    except Exception as e:
        raise CustomException(e, sys)
    
#--Sidebat - document Upload and question input --------------------------------------
# Create a sidebar in the Streamlit app where users can upload PDF files and input their questions
def render_sidebar() -> int:
    '''Render the sidebar for document upload and question input. 
    The sidebar allows users to upload PDF documents and ask questions related to the content of those documents. 
    It provides a user-friendly interface for interacting with the RAG chatbot.
    Separated from main UI for clean code organization.
    '''
    with st.sidebar:
        st.title("RAG Document Chatbot")
        st.markdown(
            """
            **Instructions:**
            1. Upload a PDF document using the uploader below.
            2. Enter a question related to the content of the uploaded document.
            3. Click the "Ask" button to get an answer based on the document's content.
            """
        )
        
        uploaded_file = st.file_uploader(
            label="Upload your PDF document",
            type=["pdf"],
            help="Choose a PDF file to upload - reports, manuals, research papers. The chatbot will use the content of this document to answer your questions."
        )
        
        #-- Handle file upload and save to temporary location for processing by the ingestor ---
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location  data/ folder for processing by the ingestor
            temp_pdf_path = os.path.join("data", uploaded_file.name)
            os.makedirs("data", exist_ok=True)
            
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # ── Save path to session state immediately ─────────────────
            # WHY: Streamlit reruns entire script on every click
            # If we don't save to session_state, temp_pdf_path disappears
            # on the next rerun when uploaded_file becomes None again
            st.session_state["temp_pdf_path"]   = temp_pdf_path
            st.session_state["uploaded_filename"] = uploaded_file.name
            
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            
            # return temp_pdf_path  # Return the path of the saved PDF for processing
        
        #Ingest button only appears after file upload - button processes the document and stores chunks in ChromaDB when clicked
        if st.button('process Document', type="primary", use_container_width=True):
            with st.spinner("Processing document... This may take a moment."):
                try:
                    
                    # -- read path from session state to ensure it persists across reruns after button click --
                    save_path = st.session_state.get("temp_pdf_path", None)
                    saved_name = st.session_state.get("uploaded_filename")
                    
                    if not save_path:
                        st.error("Please upload a PDF file first.")
                        
                    
                    #step 1: Ingest and chunk the document
                    ingestor = DataIngestion(chunk_size=500, chunk_overlap=50)
                    chunks = ingestor.load_pdf(save_path)
                    st.success(f"Document processed successfully! {len(chunks)} chunks created.")
                    
                    #step 2: Store chunks in ChromaDB
                    retriever = VectorRetriever()
                    retriever.store(chunks)
                    st.success("Chunks stored in vector database successfully!")
                    
                    #savefilename to session so main UI knows which document to query against
                    if uploaded_file is not None:
                        st.session_state["document_name"] = uploaded_file.name # Store the name of the uploaded document in session state for later use in the main UI
                        st.session_state["doc_processed"] = True  # Flag to indicate document has been processed and is ready for querying 
                        st.session_state["chunk_count"] = len(chunks)  # Store chunk count for display in main UI
                        
                        st.success(f"Ready! {len(chunks)} chunks created and stored. You can now ask questions about the document.")
                        logger.info(f"Document '{uploaded_file.name}' processed and stored successfully with {len(chunks)} chunks.")    
                    
                except Exception as e: 
                    st.error("Error processing document. Please try again." + str(e))
                    logger.error("Error processing document: " + str(e))
                    

        #show current docuemt status in sidebar
        st.divider()
        st.markdown("---")
        st.subheader("**Current Document Status**")
        if st.session_state.get("doc_processed"):
            st.success(f"Document '{st.session_state.get('document_name', 'Unknown')}' is processed and ready for querying!")
            st.caption(f"{st.session_state.get('chunk_count', 0)} chunks created from '{st.session_state.get('document_name', 'Unknown')}' and stored in vector database. You can now ask questions about this document.")
        else:
            st.warning("No document uploaded and processed yet. Please upload a PDF document and click 'Process Document' to get started.")
            st.caption("Once you upload and process a document, you can ask questions about its content in the main interface.")
            
        #setting session state variables to default values if not already set
        st.divider()
        st.markdown("**Settings:**")
        k = st.slider(
            label="Number of relevant chunks to retrieve (k)",
            min_value=1,
            max_value=20,
            value=3,
            help="Set the number of most relevant chunks to retrieve from the vector database for each question. Higher values may provide more context but can also increase response time."
        )
        return k  # Return the value of k for use in the main UI when performing searches
    
#--Main Chat Interface --------------------------------------
# Create the main chat interface where users can input their questions and receive answers based on the uploaded document
def render_chat(chain: RAGChain, k: int):
    '''Render the main chat interface for user questions and answers. This will be main chat window.
    This function creates a simple chat interface where users can input their questions related to the uploaded document and receive answers generated by the RAGChain. 
    It uses Streamlit's text input and button components to facilitate user interaction.
    '''
    st.title("🤖 RAG (Retrieval Augmented Generation) Document Chatbot")
    st.header("Ask questions about your document")
    st.caption("Powered by Phi-3 (local) + ChromaDB + LangChain")
    
    #Initialize chat history in session state to keep track of the conversation
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
                {
                    "role"    : "assistant",
                    "content" : "Hello! I'm your RAG Document Chatbot. Please upload a PDF document in the sidebar and ask me any questions related to its content. I'll do my best to provide accurate answers based on the information in the document."
                }
            ]  # List to store chat messages (questions and answers)
        
    # Display all past messages in the chat history    for message in st.session_state['messages']:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # show sources if this is an assistant message and sources are available
            if message["role"] == "assistant" and "sources" in message:
                sources = message["sources"]
                duration = message.get("duration", None)
                with st.expander(f"Sources ({len(sources)} chunks) — {duration}s" if duration else f"Sources ({len(sources)} chunks)"):
                    for i, (doc, score) in enumerate(sources):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**[{i+1}] Page {page} | Score: {score:.4f}**")
                        st.text(doc.page_content[:300].strip())
                        st.divider()
    
    #chat input box at bottom of the page for user to ask questions
    if prompt := st.chat_input("Ask a question about the document... e.g., What are the key findings in the report?"):
        
        #Block question if no document lodaded yet
        if not st.session_state.get("doc_processed"):
            st.warning("Please upload and process a PDF document in the sidebar before asking questions.")
            return
        
        # show user message in chat history
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        #generate answer using RAGChain and show in chat history
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    # Call the RAGChain to get an answer based on the user's question and the uploaded document
                    response = chain.ask(question=prompt, k=k)
                    answer   = response['answer']
                    sources  = response['sources']
                    duration = round(response['duration'], 2)
                    
                    #display answer
                    st.markdown(answer)
                    
                    # show source in expander for better UI organization
                    with st.expander(f"Sources ({len(sources)} chunks) — {duration}s"):
                        for i, (doc, score) in enumerate(sources):
                            page = doc.metadata.get("page", "?")
                            st.markdown(f"**[{i+1}] Page {page} | Score: {score:.4f}**")
                            st.text(doc.page_content[:300].strip())
                            st.divider()
                            
                    # show and save answer in chat history
                    st.session_state['messages'].append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources,
                        "duration": duration
                    })
                    
                    st.rerun()
                    logger.info(f"Question asked: '{prompt}' | Answer generated successfully.")
                    
                except Exception as e:
                    error_message = "Error generating answer. Please try again." + str(e)
                    st.error(error_message)
                    logger.error("Error generating answer: " + str(e))

#-- Entry point for the Streamlit app --------------------------------------
def main():
    try:
        #initialize session state variables for document status and chunk count
        if "doc_processed" not in st.session_state:
            st.session_state["doc_processed"] = False  # Flag to indicate if a document has been processed and is ready for querying
        
        # Load the RAGChain components once at startup
        chain = load_chain()
        
        # Render the sidebar for document upload and question input, and get the value of k for retrieval
        k = render_sidebar() or 3  # Default to 3 if render_sidebar returns None for some reason
        
        # Render the main chat interface for user questions and answers
        render_chat(chain, k)
        
    except Exception as e:
        st.error("An error occurred while running the app. Please try again." + str(e))
        logger.error("Error in main app: " + str(e))
    
if __name__ == "__main__":
    main()