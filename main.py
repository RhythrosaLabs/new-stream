import streamlit as st
from openai import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from PIL import Image
import io
import base64

# Sidebar for API keys
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")

# Titles for each model's capabilities
st.title("🌀 Unified AI Chat Interface")
st.write("Interact with AI models for general chat, file analysis, web search, and image generation.")

# Initialize session state for messages and file management
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
if "files" not in st.session_state:
    st.session_state["files"] = {}

# Display past messages
for msg in st.session_state["messages"]:
    if msg.get("image"):
        st.image(msg["image"], caption=msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# File uploader for file analysis and management
uploaded_file = st.file_uploader("Upload a file (text, image, or document)", type=["txt", "md", "pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_content = uploaded_file.read()
    st.session_state["files"][uploaded_file.name] = file_content
    st.success(f"File '{uploaded_file.name}' uploaded successfully.")

# Main command input at the bottom
prompt = st.chat_input("Type your command here...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Ensure the OpenAI API key is provided
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Analyze the prompt to determine the task
    analysis_prompt = f"Analyze the following user request to determine if it is for general chat, web search, file analysis, or image generation. Request: '{prompt}'"
    analysis_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": analysis_prompt}]
    )
    task_decision = analysis_response.choices[0].message.content.strip().lower()

    # Handle different tasks
    if "image generation" in task_decision:
        # Image generation with DALL·E 3
        try:
            image_response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1792x1024",
                quality="hd",
                style="vivid",
                response_format="b64_json"
            )
            image_data = image_response.data[0].b64_json
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            st.session_state["messages"].append({"role": "assistant", "content": f"Generated image for: '{prompt}'", "image": image})
            st.image(image, caption=f"Generated image for: '{prompt}'")
        except Exception as e:
            st.error(f"Image generation error: {e}")

    elif "file analysis" in task_decision:
        # File analysis with GPT-4 Turbo
        if uploaded_file:
            try:
                file_analysis_prompt = f"Analyze the content of the uploaded file '{uploaded_file.name}' and provide insights."
                file_analysis_response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "system", "content": file_analysis_prompt}],
                    files=[{"name": uploaded_file.name, "content": file_content}]
                )
                analysis_result = file_analysis_response.choices[0].message.content
                st.session_state["messages"].append({"role": "assistant", "content": analysis_result})
                st.chat_message("assistant").write(analysis_result)
            except Exception as e:
                st.error(f"File analysis error: {e}")
        else:
            st.warning("Please upload a file for analysis.")

    elif "web search" in task_decision:
        # Web search with LangChain's DuckDuckGo Search
        try:
            llm = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=openai_api_key, streaming=True)
            search = DuckDuckGoSearchRun(name="Search")
            search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = search_agent.run(prompt, callbacks=[st_cb])
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.write(response)
        except Exception as e:
            st.error(f"Web search error: {e}")

    else:
        # General chat response
        try:
            chat_response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=st.session_state["messages"]
            )
            chat_result = chat_response.choices[0].message.content
            st.session_state["messages"].append({"role": "assistant", "content": chat_result})
            st.chat_message("assistant").write(chat_result)
        except Exception as e:
            st.error(f"Chat response error: {e}")

# File Management System
st.subheader("File Management")

if st.session_state["files"]:
    for filename, content in st.session_state["files"].items():
        st.write(f"**{filename}**")
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(io.BytesIO(content))
            st.image(image, caption=filename)
        elif filename.lower().endswith('.txt'):
            st.text(content.decode())
        elif filename.lower().endswith('.pdf'):
            st.download_button(label="Download PDF", data=content, file_name=filename)
        else:
            st.download_button(label="Download File", data=content, file_name=filename)
else:
    st.write("No files uploaded or generated yet.")
