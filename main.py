import streamlit as st
from openai import OpenAI
import anthropic
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

# Sidebar for API keys
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

# Titles for each model's capabilities
st.title("🌀 Unified AI Chat Interface")
st.write("Interact with AI models like OpenAI, Anthropic (file Q&A), and LangChain search from a single interface.")

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

# Display past messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# File uploader for file Q&A with Anthropic
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"), key="file_uploader")

# Main command input at the bottom
if prompt := st.chat_input("Type your command here..."):
    # Append user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Check API keys
    if not (openai_api_key or anthropic_api_key):
        st.info("Please add at least one API key to continue.")
        st.stop()
    
    # OpenAI Chat
    if "openai" in prompt.lower() and openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages
        )
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
    
    # Anthropic File Q&A
    elif "anthropic" in prompt.lower() and anthropic_api_key and uploaded_file:
        article = uploaded_file.read().decode()
        question = prompt.replace("anthropic", "").strip()
        prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n{article}\n\n{question}{anthropic.AI_PROMPT}"""
        
        client = anthropic.Client(api_key=anthropic_api_key)
        response = client.completions.create(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-v1",
            max_tokens_to_sample=100,
        )
        msg = response.completion
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
    
    # LangChain with DuckDuckGo Search
    elif "search" in prompt.lower() and openai_api_key:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
        search = DuckDuckGoSearchRun(name="Search")
        search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    else:
        st.write("### Please specify 'openai', 'anthropic', or 'search' in your command to choose the model.")
