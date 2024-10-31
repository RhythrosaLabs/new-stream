import streamlit as st
import openai
import anthropic
import os
import tempfile
import requests
import base64
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import MessagesPlaceholder
import logging

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
        help="Enter your OpenAI API key. [Get one here](https://platform.openai.com/account/api-keys)",
    )
    
    # Anthropic API Key
    anthropic_api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Enter your Anthropic API key. [Get one here](https://www.anthropic.com/product/claude)",
    )
    
    st.markdown("---")
    
    # File Upload
    uploaded_file = st.file_uploader("📄 Upload a document for Q&A", type=["pdf", "txt", "md"])
    
    st.markdown("---")
    st.markdown("### Additional Tools Coming Soon!")

# ============================
# Validate API Keys
# ============================

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to use the app.")
    st.stop()

if not anthropic_api_key:
    st.warning("Please enter your Anthropic API key to enable Document Q&A.")
    # Depending on your needs, you might want to allow the app to continue without Anthropic.
    # For this example, we'll allow it to continue but disable document Q&A.

# ============================
# Set API Keys
# ============================

os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = openai_api_key

try:
    anthropic_client = anthropic.Client(anthropic_api_key) if anthropic_api_key else None
except Exception as e:
    st.error(f"Failed to initialize Anthropic client: {e}")
    anthropic_client = None

# ============================
# Initialize Session State
# ============================

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

# Image Generation Tool using OpenAI's DALL-E 3
def generate_image(prompt: str) -> str:
    """
    Generates an image based on the provided prompt using OpenAI's DALL-E 3.

    Args:
        prompt (str): The text prompt to generate the image.

    Returns:
        str: URL of the generated image or an error message.
    """
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="url"
        )
        image_url = response['data'][0]['url']
        logger.info("Image generated successfully.")
        return image_url
    except Exception as e:
        logger.exception("Exception occurred while generating image.")
        return f"Error generating image: {e}"

image_generation_tool = Tool(
    name="image_generation",
    func=generate_image,
    description="Generates an image based on the prompt. Use the command /image followed by your prompt."
)
tools.append(image_generation_tool)

# Document Q&A Tool using Anthropic's Claude
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

    if not anthropic_client:
        return "Anthropic API key not provided. Please enter it in the sidebar to use Document Q&A."

    try:
        retriever = st.session_state.vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatAnthropic(model="claude-v1", anthropic_api_key=anthropic_api_key),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        answer = qa_chain.run(question)
        return answer
    except Exception as e:
        logger.exception("Exception occurred while answering question.")
        return f"Error answering question: {e}"

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

    # Check if the prompt is an image generation command
    if prompt.strip().lower().startswith("/image"):
        image_prompt = prompt.strip()[len("/image"):].strip()
        if image_prompt:
            with st.chat_message("assistant"):
                with st.spinner("Generating image..."):
                    image_response = generate_image(image_prompt)
                    st.session_state.messages.append(AIMessage(content=image_response))
                    # Check if the response is a URL for an image
                    if image_response.startswith("http"):
                        try:
                            st.image(image_response, caption=image_prompt)
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                    else:
                        st.write(image_response)
        else:
            with st.chat_message("assistant"):
                st.write("Please provide a prompt after the /image command. Example: `/image a white siamese cat`")
    else:
        # Run the agent and get the response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                st_cb = StreamlitCallbackHandler(st.container())
                try:
                    response = agent.run(input=prompt, callbacks=[st_cb])
                except Exception as e:
                    response = f"An error occurred: {e}"
            st.session_state.messages.append(AIMessage(content=response))
            # Check if the response is an image URL
            if response.startswith("http"):
                try:
                    st.image(response, caption="Generated Image")
                except Exception as e:
                    st.error(f"Error displaying image: {e}")
            else:
                st.write(response)
