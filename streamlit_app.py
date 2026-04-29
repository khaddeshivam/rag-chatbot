import streamlit as st
import os
from dotenv import load_dotenv
import uuid

# Ensure proper SQLite usage on Linux (for Streamlit Cloud)
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from rag_utils import (
    load_doc_to_db, 
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

load_dotenv()
os.environ["USER_AGENT"] = "myagent"

def render_sidebar():
    """Render the sidebar and handle API key & model configuration."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.write("")
        with st.expander("🤖 Model Selection", expanded=True):
            provider = st.radio(
                "Select LLM Provider",
                ["OpenAI", "GROQ", "Anthropic"],
                help="Choose which Large Language Model provider to use",
                horizontal=True
            )
            if provider == "OpenAI":
                model_option = st.selectbox(
                    "Select OpenAI Model",
                    ["gpt-4o-mini", "gpt-4o", "o1", "o1-mini", "o1-preview", "o3-mini", "Custom"],
                    index=0
                )
                if model_option == "Custom":
                    model = st.text_input(
                        "Enter your custom OpenAI model:", 
                        value="", 
                        help="Specify your custom model string"
                    )
                else:
                    model = model_option
            elif provider == "GROQ":
                model = st.selectbox(
                    "Select GROQ Model",
                    [
                        "qwen-2.5-32b",
                        "deepseek-r1-distill-qwen-32b",
                        "deepseek-r1-distill-llama-70b",
                        "llama-3.3-70b-versatile",
                        "llama-3.1-8b-instant",
                        "Custom"
                    ],
                    index=0,
                    help="Choose from GROQ's available models. All these models support tool use and parallel tool use."
                )
                if model == "Custom":
                    model = st.text_input(
                        "Enter your custom GROQ model:", 
                        value="", 
                        help="Specify your custom model string"
                    )
            elif provider == "Anthropic":
                model = st.selectbox(
                    "Select Anthropic Model",
                    [
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022",
                        "claude-3-opus-20240229",
                        "claude-3-sonnet-20240229",
                        "claude-3-haiku-20240307",
                        "Custom"
                    ],
                    index=0,
                    help="Choose from Anthropic's available models. All these models support tool use and parallel tool use."
                )
                if model == "Custom":
                    model = st.text_input(
                        "Enter your custom Anthropic model:", 
                        value="", 
                        help="Specify your custom model string"
                    )
        with st.expander("🔑 API Keys", expanded=True):
            st.info("API keys are stored temporarily in memory and cleared when you close the browser.")
            if provider == "OpenAI":
                openai_api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    placeholder="Enter your OpenAI API key",
                    help="Enter your OpenAI API key"
                )
                if openai_api_key:
                    os.environ["OPENAI_API_KEY"] = openai_api_key
            elif provider == "GROQ":
                groq_api_key = st.text_input(
                    "GROQ API Key",
                    type="password",
                    placeholder="Enter your GROQ API key",
                    help="Enter your GROQ API key"
                )
                if groq_api_key:
                    os.environ["GROQ_API_KEY"] = groq_api_key
            elif provider == "Anthropic":
                anthropic_api_key = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    placeholder="Enter your Anthropic API key",
                    help="Enter your Anthropic API key"
                )
                if anthropic_api_key:
                    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
                
    return {"provider": provider, "model": model}


# --- Page Configuration & Header ---
st.set_page_config(
    page_title="Chat With Documents", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

st.html("""<h2 style="text-align: center;">📚🔍 <i> RAG LLM Chat with Documents! </i> 🤖💬</h2>""")


# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]


# Render sidebar and get selection (provider and model)
selection = render_sidebar()


# Check that API keys are set based on provider
if selection["provider"] == "OpenAI":
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to get started")
        st.stop()
elif selection["provider"] == "GROQ":
    if not os.environ.get("GROQ_API_KEY"):
        st.warning("⚠️ Please enter your GROQ API key in the sidebar to get started")
        st.stop()
elif selection["provider"] == "Anthropic":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar to get started")
        st.stop()


# --- Sidebar Additional Controls ---
with st.sidebar:
    cols0 = st.columns(2)
    with cols0[0]:
        # Enable "Use RAG" only if a vector DB exists (i.e. if a document has been loaded)
        is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
        st.toggle(
            "Use RAG", 
            value=is_vector_db_loaded, 
            key="use_rag", 
            disabled=not is_vector_db_loaded,
        )
    with cols0[1]:
        st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")
    
    st.header("RAG Sources:")
    st.file_uploader(
        "📄 Upload a document", 
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs",
    )
    st.text_input(
        "🌐 Introduce a URL", 
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
    )
    with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
        st.write([] if not is_vector_db_loaded else st.session_state.rag_sources)
    
    # Initialize the appropriate LLM stream based on the provider
    if selection["provider"] == "OpenAI":
        llm_stream = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name=selection["model"],
            temperature=0,
            streaming=True,
        )
    elif selection["provider"] == "GROQ":
        llm_stream = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model=selection["model"],
            temperature=0,
            streaming=True,
        )
    elif selection["provider"] == "Anthropic":
        llm_stream = ChatAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model=selection["model"],
            temperature=0,
            streaming=True,
        )


# --- Chat Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- Chat Input and Streaming Response ---
if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Prepare a placeholder for streaming output
        message_placeholder = st.empty()
        full_response = ""
        
        # Convert the chat history to LangChain message objects
        messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        # Choose the appropriate stream: RAG or standard LLM
        if not st.session_state.use_rag:
            st.write_stream(stream_llm_response(llm_stream, messages))
        else:
            st.write_stream(stream_llm_rag_response(llm_stream, messages))
