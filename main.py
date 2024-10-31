import streamlit as st
from openai import OpenAI
import anthropic
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

# Sidebar for API Keys
with st.sidebar:
    st.header("API Keys")
    
    # OpenAI API Key for Chatbot
    openai_api_key_chatbot = st.text_input(
        "OpenAI API Key (Chatbot)", 
        key="chatbot_api_key", 
        type="password"
    )
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

    # Anthropic API Key for File Q&A
    anthropic_api_key = st.text_input(
        "Anthropic API Key (File Q&A)", 
        key="anthropic_api_key", 
        type="password"
    )
    st.markdown("[Get an Anthropic API key](https://www.anthropic.com/account/api-keys)")

    # OpenAI API Key for LangChain Search
    openai_api_key_search = st.text_input(
        "OpenAI API Key (LangChain Search)", 
        key="langchain_search_api_key_openai", 
        type="password"
    )
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

    # GitHub Links
    st.markdown("---")
    st.markdown("[View Chatbot Source Code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)")
    st.markdown("[View File Q&A Source Code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)")
    st.markdown("[View LangChain Search Source Code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)")
    st.markdown("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)")

# Main Application
st.title("💬 Multi-Function Chat Application")

# Tabs for different functionalities
tabs = st.tabs(["OpenAI Chatbot", "Anthropic File Q&A", "LangChain Chat with Search"])

## OpenAI Chatbot Tab
with tabs[0]:
    st.header("💬 Chatbot")
    
    if "chatbot_messages" not in st.session_state:
        st.session_state["chatbot_messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    for msg in st.session_state.chatbot_messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input("Type your message here..."):
        if not openai_api_key_chatbot:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
            st.stop()
        
        try:
            client = OpenAI(api_key=openai_api_key_chatbot)
            st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=st.session_state.chatbot_messages
            )
            msg = response.choices[0].message.content
            st.session_state.chatbot_messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
        except Exception as e:
            st.error(f"An error occurred: {e}")

## Anthropic File Q&A Tab
with tabs[1]:
    st.header("📝 File Q&A with Anthropic")
    
    uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
    
    question = st.text_input(
        "Ask something about the article",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )
    
    if uploaded_file and question:
        if not anthropic_api_key:
            st.info("Please add your Anthropic API key in the sidebar to continue.")
        else:
            try:
                article = uploaded_file.read().decode()
                prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n
                {article}\n\n\n\n{question}{anthropic.AI_PROMPT}"""
            
                client = anthropic.Client(api_key=anthropic_api_key)
                response = client.completions.create(
                    prompt=prompt,
                    stop_sequences=[anthropic.HUMAN_PROMPT],
                    model="claude-2",  # Updated model name
                    max_tokens_to_sample=100,
                )
                st.write("### Answer")
                st.write(response.completion)
            except anthropic.NotFoundError as e:
                st.error(f"Model not found: {e}")
            except anthropic.AuthenticationError:
                st.error("Authentication failed. Please check your API key.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

## LangChain Chat with Search Tab
with tabs[2]:
    st.header("🔎 LangChain - Chat with Search")
    
    st.markdown("""
    In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
    Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
    """)
    
    if "search_messages" not in st.session_state:
        st.session_state["search_messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
        ]
    
    for msg in st.session_state.search_messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
        st.session_state.search_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
    
        if not openai_api_key_search:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
            st.stop()
    
        try:
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", 
                openai_api_key=openai_api_key_search, 
                streaming=True
            )
            search = DuckDuckGoSearchRun(name="Search")
            search_agent = initialize_agent(
                [search], 
                llm, 
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                handle_parsing_errors=True
            )
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = search_agent.run(st.session_state.search_messages, callbacks=[st_cb])
                st.session_state.search_messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
