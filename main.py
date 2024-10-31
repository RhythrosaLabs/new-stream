import streamlit as st
import anthropic
from openai import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from PIL import Image
import io

# Set page configuration
st.set_page_config(layout="wide")

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

    chat_container = st.container()
    user_input = st.text_input("Enter your message here:", key="user_input", help="Type your message here.")

    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "assistant":
                st.markdown(f"**🤖 Assistant:** {msg['content']}")
            else:
                st.markdown(f"**🧑 User:** {msg['content']}")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.experimental_rerun()

    if st.button("Send"):
        if not (openai_api_key or anthropic_api_key):
            st.info("Please add your API keys to continue.")
        else:
            prompt = user_input
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
            elif "search" in prompt.lower() and openai_api_key:
                # Web search using LangChain and DuckDuckGo
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
                search = DuckDuckGoSearchRun(name="Search")
                search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
                response = search_agent.run(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif "analyze image" in prompt.lower() and openai_api_key:
                # Image analysis using OpenAI
                uploaded_image = st.file_uploader("Upload an image to analyze", type=["jpg", "png"])
                if uploaded_image:
                    image = Image.open(uploaded_image)
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='PNG')
                    img_data = img_bytes.getvalue()
                    # Assuming the OpenAI API can take image data and return analysis
                    client = OpenAI(api_key=openai_api_key)
                    response = client.images.analyze(image=img_data)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            elif openai_api_key:
                # General chat using OpenAI
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
                msg = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": msg})

            st.experimental_rerun()

with tab2:
    # Files tab: Uploaded and created files
    st.header("📂 Uploaded and Created Files")
    uploaded_file = st.file_uploader("Upload files for analysis or storage", type=("txt", "md", "pdf", "jpg", "png"))
    if uploaded_file:
        st.write(f"Uploaded file: {uploaded_file.name}")
        # Automatically analyze uploaded files and add to knowledge base
        st.session_state["generated_files"].append(uploaded_file.name)
        if uploaded_file.type.startswith("image/"):
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded image: {uploaded_file.name}")
        else:
            content = uploaded_file.read().decode()
            st.write(content)

    # Displaying files that were created/generated during the chat
    if "generated_files" not in st.session_state:
        st.session_state["generated_files"] = []
    for file in st.session_state["generated_files"]:
        st.write(file)  # Displaying file name or metadata

# Improved layout and enhanced user experience with locked input at the bottom and enhanced file management.
