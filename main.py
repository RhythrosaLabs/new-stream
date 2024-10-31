import streamlit as st
from openai import OpenAI
import anthropic
from langchain_community.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# Page config and title
st.set_page_config(page_title="AI Assistant Hub", layout="wide")
st.title("🤖 AI Assistant Hub")

# Sidebar for API keys and navigation
with st.sidebar:
    st.header("API Configuration")
    openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", key="anthropic_api_key", type="password")
    
    st.markdown("---")
    st.markdown("""
    - [Get OpenAI API key](https://platform.openai.com/account/api-keys)
    - [Get Anthropic API key](https://console.anthropic.com/)
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI assistant. You can:\n1. Chat with GPT-3.5\n2. Ask questions about uploaded files\n3. Search the web\n\nHow can I help you today?"}]

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 File Q&A", "🔎 Web Search"])

# Tab 1: Basic Chat with GPT-3.5
with tab1:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if chat_prompt := st.chat_input("Send a message"):
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
            
        try:
            client = OpenAI(api_key=openai_api_key)
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            st.chat_message("user").write(chat_prompt)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.messages
            )
            msg = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Tab 2: File Q&A with Claude
with tab2:
    uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
    question = st.text_input(
        "Ask something about the article",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )
    
    if uploaded_file and question:
        if not anthropic_api_key:
            st.info("Please add your Anthropic API key to continue.")
            st.stop()
            
        try:
            article = uploaded_file.read().decode()
            prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n
            {article}\n\n\n\n{question}{anthropic.AI_PROMPT}"""
            
            client = anthropic.Client(api_key=anthropic_api_key)
            response = client.completions.create(
                prompt=prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-3-sonnet-20240229",
                max_tokens_to_sample=1000,
            )
            
            st.write("### Answer")
            st.write(response.completion)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Tab 3: Web Search with LangChain
with tab3:
    st.markdown("""
    This tab uses LangChain to perform web searches and provide informed answers.
    Ask any question that requires current information!
    """)
    
    if search_prompt := st.chat_input("Ask a question"):
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
            
        try:
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=openai_api_key,
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
                response = search_agent.run(search_prompt, callbacks=[st_cb])
                st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
