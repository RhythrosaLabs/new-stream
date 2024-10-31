import streamlit as st
import anthropic
from openai import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

# Sidebar: API key inputs and model selections
with st.sidebar:
    anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

# Main tabs: Command Line Chat, Files, API Settings
st.title("🔧 Integrated Chat Interface")
tab1, tab2 = st.tabs(["💬 Command Line Chat", "📂 Files"])

with tab1:
    # Command Line Chat: Unified chat, file analysis, generation, and web search
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I can chat, analyze files, generate files, and search the web. How can I assist you?"}
        ]

    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.write(f"🤖: {msg['content']}")
        else:
            st.write(f"🧑: {msg['content']}")

    if prompt := st.text_input("Enter your message here:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.write(f"🧑: {prompt}")

        if not (openai_api_key or anthropic_api_key):
            st.info("Please add your API keys to continue.")
            st.stop()

        # Use appropriate model based on command type
        if "analyze" in prompt.lower() and anthropic_api_key:
            # File analysis using Anthropic
            uploaded_file = st.file_uploader("Upload a file to analyze", type=("txt", "md", "pdf"))
            if uploaded_file:
                article = uploaded_file.read().decode()
                file_prompt = f"{anthropic.HUMAN_PROMPT} Here's a file:\n\n{article}\n\n{prompt}{anthropic.AI_PROMPT}"
                client = anthropic.Client(api_key=anthropic_api_key)
                response = client.completions.create(
                    prompt=file_prompt,
                    stop_sequences=[anthropic.HUMAN_PROMPT],
                    model="claude-v1",
                    max_tokens_to_sample=100,
                )
                msg = response.completion
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.write(f"🤖: {msg}")
        elif "search" in prompt.lower() and openai_api_key:
            # Web search using LangChain and DuckDuckGo
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
            search = DuckDuckGoSearchRun(name="Search")
            search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
            response = search_agent.run(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(f"🤖: {response}")
        elif openai_api_key:
            # General chat using OpenAI
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
            msg = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.write(f"🤖: {msg}")

with tab2:
    # Files tab: Uploaded and created files
    st.header("📂 Uploaded and Created Files")
    uploaded_file = st.file_uploader("Upload files for analysis or storage", type=("txt", "md", "pdf", "jpg", "png"))
    if uploaded_file:
        st.write(f"Uploaded file: {uploaded_file.name}")
        # You can add more functionality here to handle file operations

    # Displaying files that were created/generated during the chat
    if "generated_files" not in st.session_state:
        st.session_state["generated_files"] = []
    for file in st.session_state["generated_files"]:
        st.write(file)  # Displaying file name or metadata

# This structure unifies chatting, file analysis, generation, and web search while allowing for easy navigation and API management.
