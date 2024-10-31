# main.py

import streamlit as st
import openai
from config import get_api_keys
from document_processor import process_uploaded_file
from tools import (
    create_web_search_tool,
    create_openai_image_generation_tool,
    create_stability_image_generation_tool,
    create_stability_3d_generation_tool,
    create_stability_video_generation_tool,
    create_document_qa_tool
)
from agent import initialize_langchain_agent
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks import StreamlitCallbackHandler
import base64

def main():
    # Set up Streamlit page configuration
    st.set_page_config(page_title="🤖 All-in-One Chat Assistant", page_icon="🤖")

    # Get API keys and uploaded file from sidebar
    openai_api_key, stability_api_key, uploaded_file = get_api_keys()

    # Check for OpenAI API key
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to use the app.")
        st.stop()

    # Set OpenAI API key
    openai.api_key = openai_api_key

    # Initialize session state
    if "messages" not in st.session_state:
        initial_message = (
            "You are an assistant that can chat, generate images, create 3D models, generate videos, "
            "analyze documents, and search the web. You have access to various tools like OpenAI's DALL·E 3 "
            "and Stability AI's services. Use them appropriately based on the user's requests."
        )
        st.session_state.messages = [
            SystemMessage(content=initial_message)
        ]
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # If a document is uploaded, process it
    if uploaded_file:
        vectorstore, status_message = process_uploaded_file(uploaded_file, openai_api_key)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.success(status_message)
            # Update initial message to inform the assistant
            st.session_state.messages[0].content += " A document has been uploaded, and you can answer questions about it."
        else:
            st.error(status_message)

    # Define tools for the agent
    tools = [
        create_web_search_tool(),
        create_openai_image_generation_tool(openai_api_key),
        create_document_qa_tool(st.session_state.vectorstore, openai_api_key)
    ]

    # Add Stability AI tools if API key is provided
    if stability_api_key:
        tools.extend([
            create_stability_image_generation_tool(stability_api_key),
            create_stability_3d_generation_tool(stability_api_key),
            create_stability_video_generation_tool(stability_api_key)
        ])
    else:
        st.warning("Stability AI API key not provided. Stability AI features will be disabled.")

    # Initialize the agent
    agent, memory = initialize_langchain_agent(tools, openai_api_key=openai_api_key)

    # Add initial system message to agent memory
    if len(memory.chat_memory.messages) == 0:
        memory.chat_memory.add_message(st.session_state.messages[0])

    # Display chat messages from history
    st.title("🤖 All-in-One Chat Assistant")

    for msg in st.session_state.messages[1:]:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

    # Accept user input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to session state and memory
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)
        memory.chat_memory.add_message(user_message)
        st.chat_message("user").write(prompt)

        # Run the agent and get the response
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            try:
                response = agent.run(input=prompt, callbacks=[st_cb])
            except Exception as e:
                response = f"An error occurred: {e}"
            ai_message = AIMessage(content=response)
            st.session_state.messages.append(ai_message)
            memory.chat_memory.add_message(ai_message)

            # Handle different types of responses
            if response.startswith("data:image"):
                # Image response
                header, data = response.split(',', 1)
                img_data = base64.b64decode(data)
                st.image(img_data, caption="Generated Image")
            elif response.startswith("data:model"):
                # 3D Model response
                st.write("3D model generated. Currently, displaying 3D models is not supported in Streamlit.")
                # Optionally, provide a download link
                header, data = response.split(',', 1)
                glb_data = base64.b64decode(data)
                st.download_button(
                    label="Download 3D Model",
                    data=glb_data,
                    file_name="model.glb",
                    mime="model/gltf-binary"
                )
            elif response.startswith("data:video"):
                # Video response
                header, data = response.split(',', 1)
                video_data = base64.b64decode(data)
                st.video(video_data)
            else:
                # Text response
                st.write(response)

if __name__ == "__main__":
    main()
