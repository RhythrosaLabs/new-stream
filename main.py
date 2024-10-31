import streamlit as st
from openai import OpenAI
import anthropic
import requests
from datetime import datetime
import json
import time

# Page config and title
st.set_page_config(page_title="AI Assistant Hub", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px;
        gap: 12px;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e0e2e6;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Enhanced AI Assistant Hub")

# Sidebar configuration
with st.sidebar:
    st.header("🔑 API Configuration")
    openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", key="anthropic_api_key", type="password")
    
    st.markdown("---")
    
    # Model selection
    st.header("⚙️ Settings")
    openai_model = st.selectbox(
        "OpenAI Model",
        ["gpt-3.5-turbo", "gpt-4-turbo-preview"]
    )
    anthropic_model = st.selectbox(
        "Anthropic Model",
        ["claude-3-sonnet-20240229", "claude-3-opus-20240229"]
    )
    
    # Temperature settings
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    st.markdown("---")
    st.markdown("""
    ### Get API Keys
    - [OpenAI API Keys](https://platform.openai.com/account/api-keys)
    - [Anthropic API Keys](https://console.anthropic.com/)
    """)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your enhanced AI assistant. How can I help you today?"}]
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 AI Chat", 
    "📝 Document Analysis",
    "🔄 AI Model Comparison",
    "📊 Chat History"
])

# Tab 1: Enhanced AI Chat
with tab1:
    st.header("Chat with AI")
    
    # Model selector for this chat
    chat_model = st.radio(
        "Select AI Model",
        ["OpenAI GPT", "Anthropic Claude"],
        horizontal=True
    )
    
    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if chat_prompt := st.chat_input("Send a message"):
        if chat_model == "OpenAI GPT" and not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        elif chat_model == "Anthropic Claude" and not anthropic_api_key:
            st.info("Please add your Anthropic API key to continue.")
            st.stop()
            
        try:
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            st.chat_message("user").write(chat_prompt)
            
            if chat_model == "OpenAI GPT":
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model=openai_model,
                    messages=st.session_state.messages,
                    temperature=temperature
                )
                msg = response.choices[0].message.content
            else:
                client = anthropic.Client(api_key=anthropic_api_key)
                message = client.messages.create(
                    model=anthropic_model,
                    max_tokens=2000,
                    temperature=temperature,
                    messages=[{
                        "role": "user",
                        "content": chat_prompt
                    }]
                )
                msg = message.content[0].text
            
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
            
            # Save to history
            st.session_state.conversation_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": chat_model,
                "prompt": chat_prompt,
                "response": msg
            })
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Tab 2: Document Analysis
with tab2:
    st.header("Document Analysis")
    
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf", "docx"])
    if uploaded_file:
        st.session_state.current_file = uploaded_file
        
        # Display file content
        try:
            content = uploaded_file.read().decode()
            with st.expander("Document Content"):
                st.text(content)
            
            # Analysis options
            analysis_type = st.selectbox(
                "Choose analysis type",
                ["Summary", "Key Points", "Sentiment Analysis", "Custom Question"]
            )
            
            if analysis_type == "Custom Question":
                question = st.text_input("Enter your question about the document")
            else:
                question = f"Provide a {analysis_type.lower()} of this document."
            
            if st.button("Analyze"):
                if not anthropic_api_key:
                    st.info("Please add your Anthropic API key to continue.")
                    st.stop()
                
                try:
                    client = anthropic.Client(api_key=anthropic_api_key)
                    message = client.messages.create(
                        model=anthropic_model,
                        max_tokens=2000,
                        temperature=temperature,
                        messages=[{
                            "role": "user",
                            "content": f"Here's a document:\n\n{content}\n\n{question}"
                        }]
                    )
                    
                    st.write("### Analysis Results")
                    st.write(message.content[0].text)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Tab 3: AI Model Comparison
with tab3:
    st.header("AI Model Comparison")
    
    comparison_prompt = st.text_area("Enter text to compare responses across models")
    
    if st.button("Compare Models"):
        if not (openai_api_key and anthropic_api_key):
            st.info("Please add both API keys to compare models.")
            st.stop()
            
        try:
            # Create columns for comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("OpenAI GPT Response")
                with st.spinner("Generating response..."):
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model=openai_model,
                        messages=[{"role": "user", "content": comparison_prompt}],
                        temperature=temperature
                    )
                    st.write(response.choices[0].message.content)
            
            with col2:
                st.subheader("Anthropic Claude Response")
                with st.spinner("Generating response..."):
                    client = anthropic.Client(api_key=anthropic_api_key)
                    message = client.messages.create(
                        model=anthropic_model,
                        max_tokens=2000,
                        temperature=temperature,
                        messages=[{
                            "role": "user",
                            "content": comparison_prompt
                        }]
                    )
                    st.write(message.content[0].text)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Tab 4: Chat History
with tab4:
    st.header("Chat History")
    
    if st.session_state.conversation_history:
        # Add download button for chat history
        if st.button("Download Chat History"):
            history_str = json.dumps(st.session_state.conversation_history, indent=2)
            st.download_button(
                "Download JSON",
                history_str,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Display history
        for idx, entry in enumerate(reversed(st.session_state.conversation_history)):
            with st.expander(f"{entry['timestamp']} - {entry['model']}"):
                st.write("**Prompt:**", entry['prompt'])
                st.write("**Response:**", entry['response'])
    else:
        st.info("No chat history available yet.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
