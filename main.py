import streamlit as st
import openai
import requests
import os
import tempfile
import base64
import uuid
import logging
from langchain.agents import initialize_agent, Tool, AgentType
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
# Logging Configuration
# ============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        help="Enter your OpenAI API key.",
    )

    # Stability AI API Key
    stability_api_key = st.text_input(
        "Stability AI API Key",
        type="password",
        help="Enter your Stability AI API key.",
    )

    # File Upload
    uploaded_file = st.file_uploader("📄 Upload a document for Q&A", type=["pdf", "txt", "md"])

# Validate API Keys
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to use the app.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = openai_api_key

# Initialize session state for messages and document processing
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are an assistant that can chat, generate images, analyze documents, and generate media.")
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = UnstructuredFileLoader(tmp_file_path)
        documents = loader.load()

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
        try:
            os.unlink(tmp_file_path)
        except Exception as unlink_error:
            st.warning(f"Could not delete temporary file: {unlink_error}")

# ============================
# Define Tools for the Agent
# ============================

tools = []

search_tool = Tool(
    name="web_search",
    func=DuckDuckGoSearchRun().run,
    description="Answers questions about current events or the internet."
)
tools.append(search_tool)

# Image Generation Tool
def generate_image(prompt: str) -> str:
    if not stability_api_key:
        return "Stability AI API key not provided."

    url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
    headers = {
        "Authorization": f"Bearer {stability_api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": prompt,
        "output_format": "base64",
        "size": "1024x1024",
        "quality": "standard"
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            image_base64 = response.json().get("data", "")
            if image_base64:
                data_url = f"data:image/png;base64,{image_base64}"
                return data_url
            else:
                return "Error: No image data received."
        else:
            error_message = response.json().get("error", "Unknown error.")
            return f"Error generating image: {error_message}"
    except Exception as e:
        return f"Error generating image: {e}"

image_generation_tool = Tool(
    name="image_generation",
    func=generate_image,
    description="Generates images using Stability AI."
)
tools.append(image_generation_tool)

# Document Q&A Tool
def answer_question_about_document(question: str) -> str:
    if st.session_state.vectorstore is None:
        return "No document uploaded. Please upload a document."
    
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
    description="Answers questions based on the uploaded document."
)
tools.append(document_qa_tool)

# ============================
# Initialize the Agent
# ============================

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

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    if prompt.strip().lower().startswith("/image"):
        image_prompt = prompt.strip()[len("/image"):].strip()
        if image_prompt:
            with st.chat_message("assistant"):
                st.spinner("Generating image...")
                image_response = generate_image(image_prompt)
                st.session_state.messages.append(AIMessage(content=image_response))
                if image_response.startswith("data:image"):
                    try:
                        header, encoded = image_response.split(",", 1)
                        image_bytes = base64.b64decode(encoded)
                        st.image(image_bytes, caption=image_prompt)
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")
                else:
                    st.write(image_response)
        else:
            with st.chat_message("assistant"):
                st.write("Please provide a prompt after the /image command. Example: `/image a white siamese cat`")
    else:
        with st.chat_message("assistant"):
            st.spinner("Generating response...")
            try:
                response = agent.run(input=prompt)
            except Exception as e:
                response = f"An error occurred: {e}"
            st.session_state.messages.append(AIMessage(content=response))
            st.write(response)
