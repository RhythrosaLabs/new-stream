import streamlit as st
from openai import OpenAI
import anthropic
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "uploaded_file_content" not in st.session_state:
    st.session_state["uploaded_file_content"] = None

# Sidebar Configuration
with st.sidebar:
    st.header("🔑 API Keys")
    
    # OpenAI API Key
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        help="Enter your OpenAI API key."
    )
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
    
    st.markdown("---")
    
    # Anthropic API Key
    anthropic_api_key = st.text_input(
        "Anthropic API Key", 
        type="password", 
        help="Enter your Anthropic API key."
    )
    st.markdown("[Get an Anthropic API key](https://www.anthropic.com/account/api-keys)")
    
    st.markdown("---")
    
    # GitHub Links (Removed as per request)
    # Removed the "View in Codespaces" and "View on GitHub" links

# Main Application
st.title("💬 Unified Chatbot Application")

# File Uploader
st.header("📄 Upload a File for Analysis")
uploaded_file = st.file_uploader("Upload a text or markdown file", type=["txt", "md"])

if uploaded_file:
    try:
        file_content = uploaded_file.read().decode("utf-8")
        st.session_state["uploaded_file_content"] = file_content
        st.success(f"Uploaded `{uploaded_file.name}` successfully!")
    except Exception as e:
        st.error(f"❌ Failed to read the uploaded file: {e}")

# Display Chat Messages
st.header("💬 Conversation")
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

# Chat Input
prompt = st.text_input("Type your message here...", key="chat_input")

def is_search_query(query):
    search_keywords = ["search for", "look up", "find", "google", "web search", "search the web"]
    return any(keyword in query.lower() for keyword in search_keywords)

def is_file_query(query):
    file_keywords = ["analyze", "summary", "summarize", "tell me about", "information from", "details of"]
    return any(keyword in query.lower() for keyword in file_keywords)

if st.button("Send") and prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    # Determine which API to use
    if is_file_query(prompt) and st.session_state["uploaded_file_content"]:
        if not anthropic_api_key:
            response = "❌ Please provide your Anthropic API key in the sidebar to use file analysis."
        else:
            try:
                client = anthropic.Client(api_key=anthropic_api_key)
                prompt_text = f"""{anthropic.HUMAN_PROMPT} {st.session_state["uploaded_file_content"]}\n\n{prompt}{anthropic.AI_PROMPT}"""
                response_obj = client.completions.create(
                    prompt=prompt_text,
                    model="claude-2",  # Use "claude-v1" if "claude-2" is unavailable
                    stop_sequences=[anthropic.HUMAN_PROMPT],
                    max_tokens_to_sample=150,
                )
                response = response_obj.completion.strip()
            except anthropic.NotFoundError:
                response = "❌ The specified model was not found. Please check the model name."
            except anthropic.AuthenticationError:
                response = "❌ Authentication failed. Please verify your Anthropic API key."
            except Exception as e:
                response = f"❌ An unexpected error occurred: {e}"
    elif is_search_query(prompt):
        if not openai_api_key:
            response = "❌ Please provide your OpenAI API key in the sidebar to perform web searches."
        else:
            try:
                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo", 
                    openai_api_key=openai_api_key, 
                    streaming=False
                )
                search = DuckDuckGoSearchRun()
                search_agent = initialize_agent(
                    [search],
                    llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False
                )
                search_query = prompt
                search_result = search_agent.run(search_query)
                response = search_result
            except Exception as e:
                response = f"❌ An error occurred during web search: {e}"
    else:
        if not openai_api_key:
            response = "❌ Please provide your OpenAI API key in the sidebar to continue chatting."
        else:
            try:
                client = OpenAI(api_key=openai_api_key)
                messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]]
                response_obj = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                response = response_obj.choices[0].message.content.strip()
            except Exception as e:
                response = f"❌ An error occurred during chat: {e}"
    
    # Append assistant's response to messages
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # Optionally, scroll to the latest message or handle UI updates here
    # Removed st.experimental_rerun() to prevent AttributeError

    # Clear the input field by resetting the key (alternative approach)
    st.session_state["chat_input"] = ""
