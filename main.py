import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import anthropic
from openai import OpenAI

# Sidebar API Key Inputs
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="combined_api_key_openai", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", key="combined_api_key_anthropic", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("🤖 Unified LangChain, Chatbot & File Q&A Interface")

# Initial Message State
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm an assistant who can chat, answer questions about files, and search the web. How can I help you today?"}
    ]

# Displaying Chat Messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat Input
if prompt := st.chat_input(placeholder="Ask me anything or upload a file for Q&A..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Handle Missing API Key
    if not openai_api_key and not anthropic_api_key:
        st.info("Please add your OpenAI or Anthropic API key to continue.")
        st.stop()

    # OpenAI Chat Completion
    if openai_api_key:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
        search = DuckDuckGoSearchRun(name="Search")
        search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

    # Anthropic File Q&A
    if anthropic_api_key:
        client = anthropic.Client(api_key=anthropic_api_key)
        response = client.completions.create(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-v1",
            max_tokens_to_sample=100,
        )
        st.session_state.messages.append({"role": "assistant", "content": response.completion})
        st.chat_message("assistant").write(response.completion)

# File Upload Q&A
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question and anthropic_api_key:
    article = uploaded_file.read().decode()
    prompt = f"{anthropic.HUMAN_PROMPT} Here's an article:\n\n{article}\n\n\n\n{question}{anthropic.AI_PROMPT}"
    client = anthropic.Client(api_key=anthropic_api_key)
    response = client.completions.create(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=100,
    )
    st.write("### Answer")
    st.write(response.completion)

