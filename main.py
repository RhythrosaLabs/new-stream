import streamlit as st
import openai
import os
import tempfile
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import MessagesPlaceholder
from langchain.docstore.document import Document

# Set up Streamlit page configuration
st.set_page_config(page_title="All-in-One Chat Assistant", page_icon="🤖")

# Sidebar for API keys
with st.sidebar:
    st.header("🔑 API Keys")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
    st.markdown("---")
    uploaded_file = st.file_uploader("📄 Upload a document for Q&A", type=["txt", "pdf", "md"])

# Check for OpenAI API key
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to use the app.")
    st.stop()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = openai_api_key

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are an assistant that can chat, generate images, analyze documents, and search the web.")
    ]
if "document_content" not in st.session_state:
    st.session_state.document_content = ""
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# If a document is uploaded, process it
if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        from langchain.document_loaders import PyPDFLoader
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        os.unlink(tmp_file_path)  # Delete the temporary file
    else:
        # For txt and md files
        content = uploaded_file.read().decode('utf-8')
        documents = [Document(page_content=content, metadata={"source": uploaded_file.name})]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.session_state.vectorstore = vectorstore
    st.session_state.document_content = "Document uploaded and processed successfully."

# Define tools for the agent
tools = []

# Web Search Tool
search_tool = Tool(
    name="web_search",
    func=DuckDuckGoSearchRun().run,
    description="Useful for answering questions about current events or the internet."
)
tools.append(search_tool)

# Image Generation Tool
def generate_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return image_url

image_generation_tool = Tool(
    name="image_generation",
    func=generate_image,
    description="Generates an image based on the prompt."
)
tools.append(image_generation_tool)

# Document Q&A Tool
def answer_question_about_document(question):
    if st.session_state.vectorstore is None:
        return "No document has been uploaded. Please upload a document to use this feature."
    retriever = st.session_state.vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=retriever
    )
    answer = qa_chain.run(question)
    return answer

document_qa_tool = Tool(
    name="document_qa",
    func=answer_question_about_document,
    description="Useful for answering questions about the uploaded document."
)
tools.append(document_qa_tool)

# Initialize the agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")]
}
llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory,
    agent_kwargs=agent_kwargs,
)

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
        st.write(response)

        # If the response contains an image URL, display the image
        if response.startswith("http"):
            st.image(response)
