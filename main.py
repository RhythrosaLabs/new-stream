"""
All-in-One AI Assistant
=======================

A Streamlit web application that integrates OpenAI's GPT models, Anthropic's Claude,
LangChain for web search, and OpenAI's DALL·E for image generation.

Features:
- Chat with OpenAI's GPT models
- File Q&A using Anthropic's Claude
- Web Search with LangChain
- Image Generation with DALL·E

Author: Your Name
Date: 2024-10-31
"""

import streamlit as st
import openai
import anthropic
import os
import tempfile
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
st.set_page_config(
    page_title="All-in-One AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for API keys and navigation
with st.sidebar:
    st.header("🔑 API Keys")
    
    # OpenAI API Key
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key. [Get one here](https://platform.openai.com/account/api-keys)",
    )
    
    st.markdown("---")
    
    # Anthropic API Key
    anthropic_api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Enter your Anthropic API key. [Get one here](https://www.anthropic.com/product/claude)",
    )
    
    st.markdown("---")
    
    # Navigation
    st.header("📂 Navigate")
    app_mode = st.radio("Choose the app mode:", ["Chatbot", "File Q&A", "Web Search", "Image Generation"])

# Check for required API keys based on selected mode
if app_mode == "Chatbot" and not openai_api_key:
    st.warning("Please enter your OpenAI API key to use the Chatbot.")
    st.stop()

if app_mode == "File Q&A" and not anthropic_api_key:
    st.warning("Please enter your Anthropic API key to use File Q&A.")
    st.stop()

if app_mode in ["Chatbot", "Web Search", "Image Generation"] and not openai_api_key:
    st.warning("Please enter your OpenAI API key to use this feature.")
    st.stop()

# ============================
# Chatbot with OpenAI
# ============================

def chatbot_section():
    st.title("💬 Chatbot with OpenAI GPT")
    
    # Initialize session state for chatbot
    if "chatbot_messages" not in st.session_state:
        st.session_state.chatbot_messages = [
            {"role": "assistant", "content": "Hi, I'm your AI assistant. How can I help you today?"}
        ]
    
    # Display chat history
    for msg in st.session_state.chatbot_messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # User input
    if prompt := st.chat_input("Type your message here..."):
        # Append user message
        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Call OpenAI API
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # You can switch to "gpt-4" if available
                messages=st.session_state.chatbot_messages,
                stream=False,
            )
            reply = response.choices[0].message['content']
        except Exception as e:
            reply = f"Error: {e}"
        
        # Append assistant's reply
        st.session_state.chatbot_messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)

# ============================
# File Q&A with Anthropic
# ============================

def file_qa_section():
    st.title("📝 File Q&A with Anthropic Claude")
    
    uploaded_file = st.file_uploader("📄 Upload a document (PDF, TXT, MD)", type=["pdf", "txt", "md"])
    
    question = st.text_input(
        "❓ Ask a question about the uploaded document",
        placeholder="e.g., Can you summarize this article?",
        disabled=not uploaded_file,
    )
    
    if uploaded_file and question:
        if not anthropic_api_key:
            st.info("Please add your Anthropic API key to continue.")
            st.stop()
        
        # Process the uploaded file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Choose the appropriate loader
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            else:
                loader = UnstructuredFileLoader(tmp_file_path)
            
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = FAISS.from_documents(docs, embeddings)
            
            # Initialize RetrievalQA chain
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key, temperature=0),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,
            )
            
            # Generate prompt for Anthropic
            article = uploaded_file.read().decode() if uploaded_file.type != "application/pdf" else ""
            if uploaded_file.type == "application/pdf":
                # For PDFs, concatenate all text
                article = "\n".join([doc.page_content for doc in docs])
            
            anthropic_prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n{article}\n\n\n\n{question}{anthropic.AI_PROMPT}"""
            
            # Initialize Anthropic client
            anthropic_client = anthropic.Client(api_key=anthropic_api_key)
            
            # Get response from Anthropic
            response = anthropic_client.completions.create(
                prompt=anthropic_prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-v1",  # Use "claude-2" if available
                max_tokens_to_sample=300,
            )
            answer = response.completion.strip()
            
            # Display answer
            st.write("### Answer")
            st.write(answer)
        
        except Exception as e:
            st.error(f"Error processing file or generating answer: {e}")
        
        finally:
            # Delete the temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception as unlink_error:
                st.warning(f"Could not delete temporary file: {unlink_error}")

# ============================
# Web Search with LangChain
# ============================

def web_search_section():
    st.title("🔎 Chat with Web Search (LangChain)")
    
    # Initialize session state for web search
    if "web_search_messages" not in st.session_state:
        st.session_state.web_search_messages = [
            {"role": "assistant", "content": "Hi, I can help you with web searches. What do you want to know?"}
        ]
    
    # Display chat history
    for msg in st.session_state.web_search_messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # User input
    if prompt := st.chat_input("Type your question here..."):
        # Append user message
        st.session_state.web_search_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Initialize LangChain agent with DuckDuckGo search
        try:
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",  # You can switch to "gpt-4" if available
                openai_api_key=openai_api_key,
                temperature=0,
                streaming=True,
            )
            
            search = DuckDuckGoSearchRun(name="Search")
            tools = [search]
            
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                handle_parsing_errors=True,
            )
            
            # Run the agent
            callback_handler = StreamlitCallbackHandler(st.container())
            response = agent.run(prompt, callbacks=[callback_handler])
        
        except Exception as e:
            response = f"Error during web search: {e}"
        
        # Append assistant's reply
        st.session_state.web_search_messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# ============================
# Image Generation with DALL·E
# ============================

def image_generation_section():
    st.title("🖼️ Image Generation with DALL·E")
    
    # User input for image generation
    prompt = st.text_input("📥 Enter a description for the image:", "")
    size = st.selectbox("🖼️ Select image size:", ["1024x1024", "1024x1792", "1792x1024"])
    quality = st.selectbox("🎨 Select image quality:", ["standard", "hd"])
    generate_button = st.button("Generate Image")
    
    if generate_button:
        if not prompt:
            st.warning("Please enter a prompt to generate an image.")
        else:
            with st.spinner("Generating image..."):
                try:
                    response = openai.Image.create(
                        prompt=prompt,
                        size=size,
                        n=1,
                        response_format="url",
                        # Note: As of current OpenAI API, 'quality' parameter might not be supported. Adjust as necessary.
                    )
                    image_url = response['data'][0]['url']
                    st.image(image_url, caption=prompt)
                except Exception as e:
                    st.error(f"Error generating image: {e}")

# ============================
# Main App Logic
# ============================

if app_mode == "Chatbot":
    chatbot_section()
elif app_mode == "File Q&A":
    file_qa_section()
elif app_mode == "Web Search":
    web_search_section()
elif app_mode == "Image Generation":
    image_generation_section()
