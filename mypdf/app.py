import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def extract_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def main():
    st.title("Chat with PDF 💬")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # File upload
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        # Process PDF
        text = extract_text(pdf)
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        text_chunks = text_splitter.split_text(text)
        
        # Create vector store
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = create_vector_store(text_chunks)
        
        # Initialize QA chain
        llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
                           model_kwargs={"temperature":0.2, "max_length":512})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever()
        )
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Question input
        if question := st.chat_input("Ask your question"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Get answer
            with st.spinner("Thinking..."):
                response = qa_chain.run(question)
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()