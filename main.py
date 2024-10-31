"""
All-in-One Chat Assistant
=========================

A Streamlit web application that integrates OpenAI's GPT models, LangChain, and other AI tools
to provide a versatile chat assistant capable of web searching, image generation, document analysis,
and more.

Features:
- Chat with an AI assistant
- Upload and analyze documents (PDF, TXT, MD)
- Generate images based on prompts
- Perform web searches
- Retrieve answers from uploaded documents

Dependencies:
- streamlit
- openai
- langchain
- faiss-cpu
- duckduckgo-search
- python-dotenv

Author: Your Name
Date: 2024-10-31
"""

import streamlit as st
import openai
import os
import tempfile
from openai import OpenAI
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

# Set up Streamlit page configuration
st.set_page_config(page_title="All-in-One Chat Assistant", page_icon="🤖", layout="wide")

# Sidebar for API keys and file uploads
with st.sidebar:
    st.header("🔑 API Keys & Uploads")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key.")
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
    st.markdown("---")
    uploaded_file = st.file_uploader("📄 Upload a document for Q&A", type=["txt", "pdf", "md"])
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
client = OpenAI(api_key=openai_api_key)

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
        st.stop()
    finally:
        # Delete the temporary file
        os.unlink(tmp_file_path)

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

# Image Generation Tool
def generate_image(prompt: str) -> str:
    """
    Generates an image based on the provided prompt using OpenAI's DALL-E model.

    Args:
        prompt (str): The text prompt to generate the image.

    Returns:
        str: URL of the generated image or an error message.
    """
    try:
        response = openai.Image.create(
            prompt=prompt,
            model="dall-e-3",
            size="1024x1024",
            n=1,
            response_format="url"
        )
        image_url = response['data'][0]['url']
        return image_url
    except Exception as e:
        return f"Error generating image: {e}"

image_generation_tool = Tool(
    name="image_generation",
    func=generate_image,
    description="Generates an image based on the prompt."
)
tools.append(image_generation_tool)

# Document Q&A Tool
def answer_question_about_document(question: str) -> str:
    """
    Answers a question based on the uploaded document using Retrieval QA.

    Args:
        question (str): The user's question.

    Returns:
        str: The answer to the question.
    """
    if st.session_state.vectorstore is None:
        return "No document has been uploaded. Please upload a document to use this feature."
    retriever = st.session_state.vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    answer = qa_chain.run(question)
    return answer

document_qa_tool = Tool(
    name="document_qa",
    func=answer_question_about_document,
    description="Useful for answering questions about the uploaded document."
)
tools.append(document_qa_tool)

# ============================
# Initialize the Agent
# ============================

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")]
}
llm = ChatOpenAI(model_name="gpt-4", streaming=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory,
    agent_kwargs=agent_kwargs,
)

# ============================
# User Interface
# ============================

# Display chat messages from history on app rerun
st.title("🤖 All-in-One Chat Assistant")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Accept user input
if prompt := st.chat_input("Type your message here..."):
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
        # Check if the response is an image URL
        if response.startswith("http"):
            st.image(response, caption=prompt)
        else:
            st.write(response)
