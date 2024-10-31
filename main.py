import streamlit as st
from openai import OpenAI
import anthropic
from datetime import datetime
import json
import replicate
import stability_sdk

# Page config and title
st.set_page_config(page_title="AI Assistant Hub", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px; gap: 12px; padding: 10px 16px; }
    .stTabs [aria-selected="true"] { background-color: #e0e2e6; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Enhanced AI Assistant Hub")

# Sidebar configuration
with st.sidebar:
    st.header("🔑 API Configuration")
    
    # API Key inputs
    openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", key="anthropic_api_key", type="password")
    replicate_api_key = st.text_input("Replicate API Key", key="replicate_api_key", type="password")
    stability_api_key = st.text_input("Stability API Key", key="stability_api_key", type="password")
    
    # Save/Load API Keys
    if st.button("Save API Keys"):
        api_keys = {
            "openai_api_key": openai_api_key,
            "anthropic_api_key": anthropic_api_key,
            "replicate_api_key": replicate_api_key,
            "stability_api_key": stability_api_key
        }
        with open("api_keys.json", "w") as f:
            json.dump(api_keys, f)
        st.success("API Keys saved successfully!")
    
    if st.button("Load API Keys"):
        try:
            with open("api_keys.json", "r") as f:
                api_keys = json.load(f)
            openai_api_key = api_keys.get("openai_api_key", "")
            anthropic_api_key = api_keys.get("anthropic_api_key", "")
            replicate_api_key = api_keys.get("replicate_api_key", "")
            stability_api_key = api_keys.get("stability_api_key", "")
            st.success("API Keys loaded successfully!")
        except FileNotFoundError:
            st.error("No saved API Keys file found.")

    st.markdown("---")
    
    # Model selection
    st.header("⚙️ Settings")
    
    openai_models = [
        "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4-1106-preview", "gpt-4-0314", "gpt-4-0613",
        "gpt-4o", "gpt-4o-2024-08-06", "chatgpt-4o-latest", "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18", "gpt-4o-realtime-preview", "gpt-4o-audio-preview",
        "gpt-4o-realtime-preview-2024-10-01", "gpt-4o-audio-preview-2024-10-01",
        "o1-preview", "o1-preview-2024-09-12", "o1-mini", "o1-mini-2024-09-12"
    ]
    
    openai_model = st.selectbox("OpenAI Model", openai_models)
    anthropic_model = st.selectbox("Anthropic Model", ["claude-3-sonnet-20240229", "claude-3-opus-20240229"])
    
    # Temperature setting
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

# Function to check temperature support
def model_supports_temperature(model_name):
    unsupported_models = {"o1-preview", "o1-mini"}
    return model_name not in unsupported_models

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat & Document Analysis", "🔄 AI Model Comparison", "📊 Chat History"])

# Tab 1: Combined Chat and Document Analysis
with tab1:
    st.header("Chat & Document Analysis")
    
    # Model selector
    chat_model = st.radio("Select AI Model", ["OpenAI GPT", "Anthropic Claude"], horizontal=True)
    
    # Chat display
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # File upload and display content
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf", "docx"])
    if uploaded_file:
        st.session_state.current_file = uploaded_file
        content = uploaded_file.read().decode()
        with st.expander("Document Content"):
            st.text(content)
    
    # Chat input
    if chat_prompt := st.chat_input("Send a message or ask a question about the document"):
        if chat_model == "OpenAI GPT" and not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        elif chat_model == "Anthropic Claude" and not anthropic_api_key:
            st.info("Please add your Anthropic API key to continue.")
            st.stop()
        
        # Send message
        try:
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            st.chat_message("user").write(chat_prompt)
            
            # Set up question based on chat or document
            question = chat_prompt
            if uploaded_file:
                question = f"Here's a document:\n\n{content}\n\n{chat_prompt}"
            
            if chat_model == "OpenAI GPT":
                client = OpenAI(api_key=openai_api_key)
                
                # Determine whether to pass temperature based on model
                if model_supports_temperature(openai_model):
                    response = client.chat.completions.create(
                        model=openai_model,
                        messages=st.session_state.messages,
                        temperature=temperature
                    )
                else:
                    response = client.chat.completions.create(
                        model=openai_model,
                        messages=st.session_state.messages
                    )
                
                msg = response.choices[0].message.content
            else:
                client = anthropic.Client(api_key=anthropic_api_key)
                message = client.messages.create(
                    model=anthropic_model,
                    max_tokens=2000,
                    temperature=temperature,
                    messages=[{"role": "user", "content": question}]
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
# Tab 2: AI Model Comparison
with tab2:
    st.header("AI Model Comparison")
    
    comparison_prompt = st.text_area("Enter text to compare responses across models")
    
    if st.button("Compare Models"):
        if not (openai_api_key and anthropic_api_key):
            st.info("Please add both API keys to compare models.")
            st.stop()
        
        try:
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
                        messages=[{"role": "user", "content": comparison_prompt}]
                    )
                    st.write(message.content[0].text)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Tab 3: Chat History
with tab3:
    st.header("Chat History")
    
    if st.session_state.conversation_history:
        if st.button("Clear History"):
            st.session_state.conversation_history = []
            st.success("History cleared.")
        
        for record in st.session_state.conversation_history:
            with st.expander(f"Interaction on {record['timestamp']}"):
                st.write("**Model:**", record["model"])
                st.write("**Prompt:**", record["prompt"])
                st.write("**Response:**", record["response"])
    else:
        st.info("No chat history available.")
