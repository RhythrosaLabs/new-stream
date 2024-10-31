import streamlit as st
import openai
import anthropic
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import base64
import uuid

# Sidebar API Key Setup
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", type="password")
    st.markdown("[Get OpenAI Key](https://platform.openai.com/account/api-keys)")

# Set the OpenAI API key if provided
if openai_api_key:
    openai.api_key = openai_api_key

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you?"}]

# Display chat messages
st.title("AI Multi-Tool Interface")
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Process user input
user_input = st.text_input("Enter your command or question:", "")
if st.button("Send") and user_input:
    # Append user message to session
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Analyze user input to determine the required functionality
    def determine_action(prompt):
        if any(keyword in prompt.lower() for keyword in ["search", "web"]):
            return "web_search"
        elif any(keyword in prompt.lower() for keyword in ["image", "generate image"]):
            return "image_generation"
        elif any(keyword in prompt.lower() for keyword in ["summarize", "file", "document"]):
            return "file_qa"
        elif any(keyword in prompt.lower() for keyword in ["vision", "analyze image"]):
            return "vision_analysis"
        elif any(keyword in prompt.lower() for keyword in ["speak", "text to speech", "tts"]):
            return "text_to_speech"
        else:
            return "chat"  # Default action is chat

    # Call the appropriate API based on the determined action
    action = determine_action(user_input)

    # Ensure OpenAI API key is available
    if not openai_api_key:
        st.info("Please provide an OpenAI API key to continue.")
        st.stop()

    # Execute the selected action
    if action == "web_search":
        llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)
        search = DuckDuckGoSearchRun(name="Search")
        search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        response = search_agent.run(st.session_state["messages"])
        response_content = response

    elif action == "image_generation":
        image_response = openai.Image.create(
            prompt=user_input,
            n=1,
            model="dall-e-3"
        )
        image_url = image_response['data'][0]['url']
        response_content = f"![Generated Image]({image_url})"

    elif action == "file_qa":
        if anthropic_api_key:
            uploaded_file = st.file_uploader("Upload a file for Q&A", type=["txt", "md"])
            if uploaded_file:
                question = user_input.replace("summarize ", "").replace("file", "").strip()
                article = uploaded_file.read().decode()
                anthropic_client = anthropic.Client(api_key=anthropic_api_key)
                anthropic_prompt = f"{anthropic.HUMAN_PROMPT} {question}\n\n{article}\n{anthropic.AI_PROMPT}"
                response = anthropic_client.completions.create(
                    prompt=anthropic_prompt,
                    model="claude-v1",
                    max_tokens_to_sample=100
                )
                response_content = response.completion
            else:
                response_content = "Please upload a file to ask questions about."

    elif action == "vision_analysis":
        uploaded_image = st.file_uploader("Upload an image for analysis", type=["jpg", "png"])
        if uploaded_image:
            base64_image = base64.b64encode(uploaded_image.read()).decode()
            vision_prompt = "What is shown in this image?"
            vision_response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[{"role": "user", "content": vision_prompt}]
            )
            response_content = vision_response['choices'][0]['message']['content']
        else:
            response_content = "Please upload an image for vision analysis."

    elif action == "text_to_speech":
        tts_response = openai.TextToSpeech.create(
            text=user_input,
            voice="female",
            model="tts-1-hd"
        )
        audio_url = tts_response['data'][0]['audio_url']
        response_content = f"[Generated Speech]({audio_url})"

    else:  # Default to OpenAI Chat
        chat_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=st.session_state["messages"]
        )
        response_content = chat_response['choices'][0]['message']['content']

    # Append assistant's response to the session state
    st.session_state["messages"].append({"role": "assistant", "content": response_content})
    st.chat_message("assistant").write(response_content)
