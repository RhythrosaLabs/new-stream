"""
All-in-One AI Assistant
========================

Integrates:
- OpenAI's GPT models for general chat.
- Anthropic's Claude for File Q&A.
- LangChain for Web Search.
- Stability AI's Stable Image Ultra for Image Generation.

Features:
- Single chat interface to interact with all functionalities.
- Upload documents (PDF, TXT, MD) for Q&A.
- Generate images based on text prompts.
- Perform web searches for up-to-date information.

Author: Your Name
Date: 2024-10-31
"""

import streamlit as st
import openai
import anthropic
import os
import tempfile
import requests
import base64
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import MessagesPlaceholder

# ============================
# Configuration and Setup
# ============================

st.set_page_config(
    page_title="All-in-One AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for API keys and file uploads
with st.sidebar:
    st.header("🔑 API Keys & Uploads")
    
    # OpenAI API Key
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key. [Get one here](https://platform.openai.com/account/api-keys)",
    )
    
    # Anthropic API Key
    anthropic_api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Enter your Anthropic API key. [Get one here](https://www.anthropic.com/product/claude)",
    )
    
    # Stability AI API Key
    stability_api_key = st.text_input(
        "Stability AI API Key",
        type="password",
        help="Enter your Stability AI API key. [Get one here](https://platform.stability.ai/account/api-keys)",
    )
    
    st.markdown("---")
    
    # File Upload
    uploaded_file = st.file_uploader("📄 Upload a document for Q&A", type=["pdf", "txt", "md"])
    
    st.markdown("---")
    st.markdown("### Additional Tools Coming Soon!")

# Check for OpenAI API key
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to use the app.")
    st.stop()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = openai_api_key

# Initialize OpenAI client
client = openai

# Initialize session state for messages and document processing
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are an assistant that can chat, generate images, analyze documents, and search the web.")
    ]
if "document_content" not in st.session_state:
    st.session_state.document_content = ""
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ============================
# Document Upload and Processing
# ============================

if uploaded_file:
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Choose the appropriate loader based on file type
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = UnstructuredFileLoader(tmp_file_path)
        documents = loader.load()

        # Split documents and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.session_state.vectorstore = vectorstore
        st.session_state.document_content = "Document uploaded and processed successfully."
        st.success(st.session_state.document_content)
    except Exception as e:
        st.error(f"Error processing document: {e}")
    finally:
        # Delete the temporary file
        try:
            os.unlink(tmp_file_path)
        except Exception as unlink_error:
            st.warning(f"Could not delete temporary file: {unlink_error}")

# ============================
# Define Tools for the Agent
# ============================

tools = []

# Web Search Tool
search_tool = Tool(
    name="web_search",
    func=DuckDuckGoSearchRun().run,
    description="Useful for answering questions about current events or the internet."
)
tools.append(search_tool)

# Image Generation Tool using Stability AI's Stable Image Ultra
def generate_image(prompt: str) -> str:
    """
    Generates an image based on the provided prompt using Stability AI's Stable Image Ultra.

    Args:
        prompt (str): The text prompt to generate the image.

    Returns:
        str: Data URL of the generated image or an error message.
    """
    if not stability_api_key:
        return "Stability AI API key not provided."

    url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
    headers = {
        "authorization": f"Bearer {stability_api_key}",
        "accept": "image/png"  # Options: "image/jpeg", "image/webp"
    }
    files = {
        "none": ''  # As per API documentation
    }
    data = {
        "prompt": prompt,
        "output_format": "png",  # Options: "jpeg", "png", "webp"
        "size": "1024x1024",     # Options: "1024x1024", "1024x1792", "1792x1024"
        "quality": "standard"    # Options: "standard", "hd"
    }
    
    try:
        response = requests.post(
            url,
            headers=headers,
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            image_bytes = response.content
            encoded_image = base64.b64encode(image_bytes).decode()
            data_url = f"data:image/png;base64,{encoded_image}"
            return data_url
        else:
            error_message = response.json().get('error', 'Unknown error occurred.')
            return f"Error generating image: {error_message}"
    except Exception as e:
        return f"Error generating image: {e}"

image_generation_tool = Tool(
    name="image_generation",
    func=generate_image,
    description="Generates an image based on the prompt using Stability AI's Stable Image Ultra."
)
tools.append(image_generation_tool)

# Document Q&A Tool using Anthropic
def answer_question_about_document(question: str) -> str:
    """
    Answers a question based on the uploaded document using Anthropic's Claude.

    Args:
        question (str): The user's question.

    Returns:
        str: The answer to the question.
    """
    if st.session_state.vectorstore is None:
        return "No document has been uploaded. Please upload a document to use this feature."
    
    try:
        retriever = st.session_state.vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        answer = qa_chain.run(question)
        return answer
    except Exception as e:
        return f"Error answering question: {e}"

document_qa_tool = Tool(
    name="document_qa",
    func=answer_question_about_document,
    description="Useful for answering questions about the uploaded document."
)
tools.append(document_qa_tool)

# Initialize the agent with all tools
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")]
}

llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key, streaming=True)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=False,
    memory=memory,
    agent_kwargs=agent_kwargs,
)

# ============================
# User Interface
# ============================

st.title("🤖 All-in-One AI Assistant")

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Add user message to session state
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Run the agent and get the response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(input=prompt, callbacks=[st_cb])
        except Exception as e:
            response = f"An error occurred: {e}"
        st.session_state.messages.append(AIMessage(content=response))
        # Check if the response is a data URL for an image
        if response.startswith("data:image"):
            # Extract the base64 part and decode it
            header, encoded = response.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            st.image(image_bytes, caption="Generated Image")
        else:
            st.write(response)
