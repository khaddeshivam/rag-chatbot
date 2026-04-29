import os
import shutil
from dotenv import load_dotenv
from time import time
import streamlit as st
import chromadb
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 10

def stream_llm_response(llm_stream, messages):
    response_message = ""
    
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})

### Indexing Process ###

def initialize_vector_db(docs):
    embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    
    # Create a unique persist directory based on session_id
    persist_directory = f"./chroma_db_{st.session_state['session_id']}"
    
    # Clear any existing persisted database to force a fresh initialization.
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_{st.session_state['session_id']}",
        persist_directory=persist_directory,
    )
    
    chroma_client = vector_db._client
    # list_collections() returns collection names as strings
    collection_names = sorted(chroma_client.list_collections())
    print("Number of collections:", len(collection_names))
    
    # Remove old collections if there are too many
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)
    
    return vector_db
    

def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )
    
    # Split documents into chunks
    document_chunks = text_splitter.split_documents(docs)
    # Filter out chunks with empty content
    document_chunks = [chunk for chunk in document_chunks if chunk.page_content.strip()]
    
    # Check if any valid chunks exist before indexing
    if not document_chunks:
        st.error("No valid document content found after splitting. Please check your input documents.")
        return
    
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(document_chunks)
    else:
        st.session_state.vector_db.add_documents(document_chunks)
    

def load_doc_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        # Ensure rag_sources exists in session state.
        if "rag_sources" not in st.session_state:
            st.session_state.rag_sources = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())
                    
                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue
                        
                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)
                    
                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")

                    finally:
                        os.remove(file_path)
                else:
                    st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")
        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✅")

def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if "rag_sources" not in st.session_state:
            st.session_state.rag_sources = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)
                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")
                
                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")
            else:
                st.error("Maximum number of documents reached (10).")

### End of Indexing Process ###

### Retrieval Augmented Generation (RAG) Process ###

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages."),
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful assistant. You will have to answer user's queries.
        You will have some context to help with your answers, though it might not always be completely related or sufficient.
        You can also use your general knowledge to assist in answering the queries.\n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def stream_llm_rag_response(llm_stream, messages):
    # Initialize the response message
    response_message = ""
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    
    # Stream the answer chunks, concatenating them to form the full response
    for chunk in conversation_rag_chain.pick("answer").stream({
        "messages": messages[:-1],
        "input": messages[-1].content
    }):
        # If the chunk has a 'content' attribute, use it; otherwise, assume it's a string.
        content = chunk.content if hasattr(chunk, "content") else chunk
        response_message += content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})
