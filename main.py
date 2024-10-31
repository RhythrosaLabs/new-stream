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

    # Ensure at least one API key is provided
    if not (openai_api_key or anthropic_api_key):
        st.info("Please add at least one API key to continue.")
        st.stop()
    
    # Use GPT-4o-mini to analyze and determine the required action
    client = OpenAI(api_key=openai_api_key)
    analysis_prompt = f"Analyze the following user request to determine the appropriate task. Determine if it's a general chat, a web search, file analysis, or image generation request. Here is the request:\n\n'{prompt}'"
    analysis_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": analysis_prompt}]
    )
    model_decision = analysis_response.choices[0].message.content.strip().lower()

    # OpenAI Chat (General Chat)
    if "general chat" in model_decision and openai_api_key:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages
        )
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
    
    # Anthropic File Q&A with Error Handling
    elif "file analysis" in model_decision and anthropic_api_key and uploaded_file:
        try:
            article = uploaded_file.read().decode()
            question = prompt
            anthropic_prompt = f"{anthropic.HUMAN_PROMPT} Here's an article:\n\n{article}\n\n{question}{anthropic.AI_PROMPT}"
            
            client = anthropic.Client(api_key=anthropic_api_key)
            response = client.completions.create(
                prompt=anthropic_prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-v1",
                max_tokens_to_sample=100,
            )
            msg = response.completion
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
        
        except anthropic.NotFoundError:
            st.error("Anthropic API endpoint or key issue. Please double-check your Anthropic API key.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    # LangChain with DuckDuckGo Search (Web Search)
    elif "web search" in model_decision and openai_api_key:
        llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key, streaming=True)
        search = DuckDuckGoSearchRun(name="Search")
        search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    
    # OpenAI DALL-E Image Generation
    elif "image generation" in model_decision and openai_api_key:
        try:
            image_response = client.images.create(
                prompt=prompt,
                n=1,
                size="1024x1024",
            )
            image_url = image_response['data'][0]['url']
            st.image(image_url, caption="Generated Image")
            st.session_state.messages.append({"role": "assistant", "content": f"Image generated based on: '{prompt}'"})
        
        except Exception as e:
            st.error(f"An error occurred with image generation: {e}")

    else:
        st.write("### Unable to determine the action based on the prompt. Please ensure the request is clear.")
