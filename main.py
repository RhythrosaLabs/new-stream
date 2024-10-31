import streamlit as st
from openai import OpenAI
import anthropic
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import os
import uuid

# Set page configuration
st.set_page_config(
    page_title="Unified Chat Interface",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for messages and file management
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I can help you with chatting, file analysis, and web searches. How can I assist you today?"}]
if "files" not in st.session_state:
    st.session_state["files"] = {}  # Dictionary to store files with unique IDs

# Sidebar for API keys and links
with st.sidebar:
    st.header("🔑 API Keys")
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
    anthropic_api_key = st.text_input("Anthropic API Key", type="password", key="anthropic_api_key")
    st.markdown("---")
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
    st.markdown("[Get an Anthropic API key](https://www.anthropic.com/product/claude)")
    st.markdown("[View the source code](https://github.com/your-repo/your-app)")
    st.markdown("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new)")

# Main layout with two columns: File Management and Chat Interface
col1, col2 = st.columns([1, 3])

# File Management Section
with col1:
    st.header("📂 File Management")
    
    # File uploader
    uploaded_files = st.file_uploader("Upload Files", type=["txt", "md", "py", "json", "csv"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_id = str(uuid.uuid4())
            st.session_state["files"][file_id] = {
                "name": uploaded_file.name,
                "content": uploaded_file.read(),
                "type": uploaded_file.type
            }
        st.success("Files uploaded successfully!")

    # Display list of files
    if st.session_state["files"]:
        st.subheader("Your Files")
        for file_id, file_info in st.session_state["files"].items():
            col_file, col_actions = st.columns([3, 1])
            with col_file:
                st.markdown(f"**{file_info['name']}**")
            with col_actions:
                btn_download = st.button("Download", key=f"download_{file_id}")
                btn_delete = st.button("Delete", key=f"delete_{file_id}")
                if btn_download:
                    st.download_button(
                        label="Download",
                        data=file_info["content"],
                        file_name=file_info["name"],
                        mime=file_info["type"],
                    )
                if btn_delete:
                    del st.session_state["files"][file_id]
                    st.success(f"Deleted {file_info['name']}")
    
    st.markdown("---")
    st.subheader("Generated Code Files")
    # Display generated code files
    for file_id, file_info in st.session_state["files"].items():
        if file_info["type"] in ["text/plain", "application/json", "application/python"]:
            col_file, col_actions = st.columns([3, 1])
            with col_file:
                st.markdown(f"**{file_info['name']}**")
            with col_actions:
                btn_download = st.button("Download", key=f"download_gen_{file_id}")
                btn_delete = st.button("Delete", key=f"delete_gen_{file_id}")
                if btn_download:
                    st.download_button(
                        label="Download",
                        data=file_info["content"],
                        file_name=file_info["name"],
                        mime=file_info["type"],
                    )
                if btn_delete:
                    del st.session_state["files"][file_id]
                    st.success(f"Deleted {file_info['name']}")

# Chat Interface Section
with col2:
    st.header("💬 Unified Chatbot Interface")
    
    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
        else:
            st.chat_message("user").write(msg["content"])
    
    # Chat input
    prompt = st.chat_input(placeholder="Type your message here...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Determine the type of request based on user input
        # For simplicity, using keywords to determine functionality
        if any(keyword in prompt.lower() for keyword in ["summary", "analyze", "file", "document"]):
            # Handle File Q&A with Anthropic
            if not anthropic_api_key:
                st.info("Please add your Anthropic API key to continue.")
            else:
                # Check if a file is uploaded
                if st.session_state["files"]:
                    # For simplicity, take the first uploaded file
                    file_id, file_info = next(iter(st.session_state["files"].items()))
                    article = file_info["content"].decode()
                    anthropic_client = anthropic.Client(api_key=anthropic_api_key)
                    anthropic_prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n{article}\n\n\n\n{prompt}{anthropic.AI_PROMPT}"""
                    try:
                        response = anthropic_client.completions.create(
                            prompt=anthropic_prompt,
                            stop_sequences=[anthropic.HUMAN_PROMPT],
                            model="claude-v1",
                            max_tokens_to_sample=300,
                        )
                        answer = response.completion.strip()
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.chat_message("assistant").write(answer)
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                        st.chat_message("assistant").write(f"Error: {str(e)}")
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "Please upload a file first to analyze."})
                    st.chat_message("assistant").write("Please upload a file first to analyze.")
        elif any(keyword in prompt.lower() for keyword in ["search", "find", "web"]):
            # Handle Web Search with LangChain
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
            else:
                try:
                    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
                    search = DuckDuckGoSearchRun(name="Search")
                    search_agent = initialize_agent(
                        [search],
                        llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        handle_parsing_errors=True
                    )
                    with st.chat_message("assistant"):
                        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                        response = search_agent.run(prompt, callbacks=[st_cb])
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.write(response)
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    st.chat_message("assistant").write(f"Error: {str(e)}")
        else:
            # Handle regular chat with OpenAI
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
            else:
                try:
                    openai_client = OpenAI(api_key=openai_api_key)
                    response = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state["messages"]
                    )
                    msg = response.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    st.chat_message("assistant").write(msg)
                    
                    # If the assistant provides code, save it to file management
                    if "```" in msg:
                        code = msg.split("```")[1]
                        code_language = msg.split("```")[0].split()[-1] if "```" in msg else "py"
                        filename = f"generated_code_{uuid.uuid4().hex[:8]}.{code_language}"
                        st.session_state["files"][str(uuid.uuid4())] = {
                            "name": filename,
                            "content": code.encode(),
                            "type": f"application/{code_language}"
                        }
                        st.success(f"Generated code saved as {filename}")
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    st.chat_message("assistant").write(f"Error: {str(e)}")
