import streamlit as st
from openai import OpenAI
import anthropic
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import logging

# API Setup Sidebar
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", type="password")
    st.markdown("[Get OpenAI Key](https://platform.openai.com/account/api-keys)")
    st.markdown("[Code on GitHub](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)")

# Initialize session state for messages and files
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you?"}]
if "files" not in st.session_state:
    st.session_state["files"] = []

# Chat Interface and Message Display
st.title("🧠 AI Multi-Tool Chat Interface")

def display_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

display_messages()

# Message Input Handling
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please provide an OpenAI API key to proceed.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # OpenAI Chat
    client = OpenAI(api_key=openai_api_key)
    response = client.Chat.create(model="gpt-3.5-turbo", messages=st.session_state["messages"])
    msg_content = response.choices[0].message.content
    st.session_state["messages"].append({"role": "assistant", "content": msg_content})
    st.chat_message("assistant").write(msg_content)

    # Additional functionalities like file rendering, if available
    if "files" in st.session_state:
        for file in st.session_state["files"]:
            st.write(f"File: {file['name']} - {file['type']}")
            # Custom file handling logic based on file type

# Search with LangChain Integration
def chat_with_search():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    response = search_agent.run(st.session_state["messages"])
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

st.button("🔍 Search Web", on_click=chat_with_search)

# File Upload and Q&A with Anthropic Integration
uploaded_file = st.file_uploader("Upload a file for Q&A")
question = st.text_input("Ask a question about the uploaded file", placeholder="Summarize the file")

if uploaded_file and question:
    if not anthropic_api_key:
        st.info("Please provide an Anthropic API key.")
        st.stop()
    article = uploaded_file.read().decode()
    client = anthropic.Client(api_key=anthropic_api_key)
    prompt = f"{anthropic.HUMAN_PROMPT} {question}\n\n{article}\n{anthropic.AI_PROMPT}"
    response = client.completions.create(prompt=prompt, model="claude-v1", max_tokens_to_sample=100)
    st.session_state["messages"].append({"role": "assistant", "content": response.completion})
    st.chat_message("assistant").write(response.completion)
