import streamlit as st
import os
import requests
import threading
import time
import base64
from io import BytesIO
from PIL import Image
import json
import streamlit_drawable_canvas as st_canvas

# Directory to save generated files
GENERATED_FILES_DIR = "generated_files"
if not os.path.exists(GENERATED_FILES_DIR):
    os.makedirs(GENERATED_FILES_DIR)

# Custom CSS for Chat Interface
st.markdown("""
    <style>
        /* Container to hold chat messages and make it scrollable */
        .chat-container {
            height: 70vh;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 90px; /* Space for fixed input area */
        }
        /* Fixed input area at the bottom */
        .fixed-input {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            max-width: 800px;
            padding: 10px;
            background-color: white;
            border-top: 1px solid #ddd;
            display: flex;
            align-items: center;
            z-index: 1000;
        }
        /* Adjust main content to avoid overlap with the fixed input */
        .main-content {
            padding-bottom: 90px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for API Key Configuration
with st.sidebar:
    st.header("🔑 API Configuration")
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    stability_api_key = st.text_input("Stability AI API Key", key="stability_ai_api_key", type="password")
    st.markdown("**Note:** Ensure your API keys are valid to access all features.")

# Titles for the application
st.title("🌀 Unified AI Chat Interface")

# Initialize session state for messages and file management
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
if "files" not in st.session_state:
    st.session_state["files"] = {}

# Tabs for Chat and File Management
tab1, tab2 = st.tabs(["💬 Chat", "📁 File Management"])

# --- Helper Functions ---
# Function to save generated files
def save_file(content, filename):
    file_path = os.path.join(GENERATED_FILES_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path

# Function to display images
def display_image(image_bytes, caption=None):
    image = Image.open(BytesIO(image_bytes))
    st.image(image, caption=caption, use_column_width=True)

# Function to display videos
def display_video(video_bytes, caption=None):
    video_encoded = base64.b64encode(video_bytes).decode()
    video_html = f"""
    <video width="100%" controls>
        <source src="data:video/mp4;base64,{video_encoded}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)
    if caption:
        st.caption(caption)

# Function to display 3D models using model-viewer
def display_3d_model(glb_bytes, caption=None):
    glb_base64 = base64.b64encode(glb_bytes).decode()
    model_html = f"""
    <model-viewer src="data:model/gltf-binary;base64,{glb_base64}"
                  style="width: 100%; height: 600px;"
                  autoplay
                  camera-controls
                  ar>
    </model-viewer>
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    """
    st.markdown(model_html, unsafe_allow_html=True)
    if caption:
        st.caption(caption)

# --- Tab 1: Chat ---
with tab1:
    st.write("Interact with AI models for general chat, file analysis, web search, image generation, video creation, and 3D modeling.")

    # Display chat messages in a scrollable container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state["messages"]:
        if msg.get("image"):
            display_image(msg["image"], caption=msg["content"])
        elif msg.get("video"):
            display_video(msg["video"], caption=msg["content"])
        elif msg.get("model_3d"):
            display_3d_model(msg["model_3d"], caption=msg["content"])
        else:
            st.chat_message(msg["role"]).write(msg["content"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Fixed input area for file upload and text input at the bottom of the page
    st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
    with st.form(key='chat_form', clear_on_submit=True):
        uploaded_file = st.file_uploader("Attach file", type=["txt", "md", "pdf", "png", "jpg", "jpeg"], label_visibility="collapsed")
        prompt = st.text_input("Type your command here...", key="chat_input", label_visibility="collapsed")
        submit = st.form_submit_button("Send")
    st.markdown('</div>', unsafe_allow_html=True)

    # If a file is uploaded, add it to the file management system
    if uploaded_file:
        file_content = uploaded_file.read()
        st.session_state["files"][uploaded_file.name] = file_content
        st.session_state["messages"].append({"role": "assistant", "content": f"📁 File '{uploaded_file.name}' uploaded successfully."})
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")

    if submit and prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Ensure both API keys are provided
        if not openai_api_key or not stability_api_key:
            st.error("Please enter both OpenAI and Stability AI API keys in the sidebar to continue.")
        else:
            # Initialize headers
            headers_openai = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            }
            headers_stability = {
                "Authorization": f"Bearer {stability_api_key}",
                "Content-Type": "application/json"
            }

            # Analyze the prompt to determine the task
            analysis_prompt = f"Analyze the following user request to determine if it is for general chat, web search, file analysis, image generation, video generation, or 3D model generation. Request: '{prompt}'"

            analysis_payload = {
                "model": "gpt-4-turbo",
                "messages": [{"role": "system", "content": analysis_prompt}],
                "max_tokens": 50,
                "temperature": 0.5
            }

            def analyze_prompt():
                try:
                    analysis_response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers_openai,
                        json=analysis_payload
                    )
                    if analysis_response.status_code == 200:
                        analysis_result = analysis_response.json()['choices'][0]['message']['content'].strip().lower()
                        handle_task(analysis_result)
                    else:
                        st.session_state["messages"].append({"role": "assistant", "content": f"❌ Error analyzing prompt: {analysis_response.status_code} - {analysis_response.text}"})
                except Exception as e:
                    st.session_state["messages"].append({"role": "assistant", "content": f"❌ Error analyzing prompt: {e}"})

            def handle_task(task_decision):
                if "image generation" in task_decision:
                    generate_image(prompt)
                elif "file analysis" in task_decision:
                    analyze_file(prompt)
                elif "web search" in task_decision:
                    perform_web_search(prompt)
                elif "video generation" in task_decision:
                    generate_video(prompt)
                elif "3d model generation" in task_decision:
                    generate_3d_model(prompt)
                else:
                    general_chat(prompt)

            def generate_image(prompt_text):
                def task():
                    st.session_state["messages"].append({"role": "assistant", "content": "🖼️ Generating image..."})
                    st.experimental_rerun()

                    # Prepare data for Stability AI
                    data = {
                        "prompt": prompt_text,
                        "model": "stable-diffusion-3.0-large",
                        "steps": 50,
                        "sampler": "DDIM",
                        "cfg_scale": 7.0,
                        "seed": 0,
                        "output_format": "png",
                        "mode": "text-to-image",
                        "strength": 0.5
                    }

                    url = "https://api.stability.ai/v2beta/stable-image/generate"

                    try:
                        response = requests.post(url, headers=headers_stability, json=data)
                        if response.status_code == 200:
                            data_response = response.json()
                            if 'artifacts' in data_response and len(data_response['artifacts']) > 0:
                                img_data = base64.b64decode(data_response['artifacts'][0]['base64'])
                                filename = f"text_to_image_{int(time.time())}.{data['output_format']}"
                                save_file(img_data, filename)
                                st.session_state["messages"].append({"role": "assistant", "content": f"🖼️ Image generated for: '{prompt_text}'", "image": img_data})
                            else:
                                st.session_state["messages"].append({"role": "assistant", "content": "❌ No artifacts found in the response."})
                        else:
                            st.session_state["messages"].append({"role": "assistant", "content": f"❌ Error: {response.status_code} - {response.text}"})
                    except Exception as e:
                        st.session_state["messages"].append({"role": "assistant", "content": f"❌ Image generation error: {e}"})

                threading.Thread(target=task).start()

            def analyze_file(prompt_text):
                def task():
                    if uploaded_file:
                        st.session_state["messages"].append({"role": "assistant", "content": "📄 Analyzing the uploaded file..."})
                        st.experimental_rerun()

                        file_content = st.session_state["files"][uploaded_file.name]
                        # Assuming the file is text-based for simplicity
                        if uploaded_file.type.startswith("text/"):
                            file_text = file_content.decode()
                        elif uploaded_file.type.startswith("image/"):
                            # Convert image to base64 string
                            file_text = base64.b64encode(file_content).decode()
                        else:
                            file_text = "Unsupported file type for analysis."

                        analysis_payload = {
                            "model": "gpt-4-turbo",
                            "messages": [
                                {"role": "system", "content": f"Analyze the content of the uploaded file '{uploaded_file.name}' and provide insights."},
                                {"role": "user", "content": f"File content: {file_text}"}
                            ],
                            "max_tokens": 500,
                            "temperature": 0.5
                        }

                        try:
                            response = requests.post(
                                "https://api.openai.com/v1/chat/completions",
                                headers=headers_openai,
                                json=analysis_payload
                            )
                            if response.status_code == 200:
                                analysis_result = response.json()['choices'][0]['message']['content'].strip()
                                st.session_state["messages"].append({"role": "assistant", "content": analysis_result})
                            else:
                                st.session_state["messages"].append({"role": "assistant", "content": f"❌ Error: {response.status_code} - {response.text}"})
                        except Exception as e:
                            st.session_state["messages"].append({"role": "assistant", "content": f"❌ File analysis error: {e}"})
                    else:
                        st.session_state["messages"].append({"role": "assistant", "content": "❌ Please upload a file for analysis."})

                threading.Thread(target=task).start()

            def perform_web_search(prompt_text):
                def task():
                    st.session_state["messages"].append({"role": "assistant", "content": "🔍 Performing web search..."})
                    st.experimental_rerun()

                    # Placeholder for actual web search implementation
                    # You can integrate LangChain's DuckDuckGoSearchRun or any other search API here
                    search_result = f"🌐 Search results for '{prompt_text}' are not implemented yet."
                    st.session_state["messages"].append({"role": "assistant", "content": search_result})

                threading.Thread(target=task).start()

            def generate_video(prompt_text):
                def task():
                    st.session_state["messages"].append({"role": "assistant", "content": "🎞️ Generating video..."})
                    st.experimental_rerun()

                    # Placeholder parameters; in a real scenario, parse from prompt or use defaults
                    data = {
                        "cfg_scale": 1.8,
                        "motion_bucket_id": 127,
                        "seed": 0
                    }

                    url = "https://api.stability.ai/v2beta/image-to-video"

                    if uploaded_file and uploaded_file.type.startswith("image/"):
                        try:
                            files = {
                                "image": uploaded_file.read(),
                            }

                            response = requests.post(url, headers=headers_stability, files=files, data=data)
                            if response.status_code == 200:
                                generation_id = response.json().get("id")
                                st.session_state["messages"].append({"role": "assistant", "content": "⏳ Video generation started. Please wait..."})
                                st.experimental_rerun()

                                # Polling for video generation result
                                result_url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"
                                accept_header = "video/*"
                                max_retries = 30
                                retry_delay = 10  # seconds

                                for attempt in range(max_retries):
                                    time.sleep(retry_delay)
                                    result_response = requests.get(
                                        result_url,
                                        headers={
                                            "Authorization": f"Bearer {stability_api_key}",
                                            "Accept": accept_header,
                                        },
                                    )
                                    if result_response.status_code == 200:
                                        video_bytes = result_response.content
                                        filename = f"generated_video_{int(time.time())}.mp4"
                                        save_file(video_bytes, filename)
                                        st.session_state["messages"].append({"role": "assistant", "content": "🎞️ Video generated successfully!", "video": video_bytes})
                                        break
                                    elif result_response.status_code == 202:
                                        st.session_state["messages"].append({"role": "assistant", "content": f"⏳ Still processing... ({attempt + 1}/{max_retries})"})
                                        st.experimental_rerun()
                                        continue
                                    else:
                                        st.session_state["messages"].append({"role": "assistant", "content": f"❌ Error: {result_response.status_code} - {result_response.text}"})
                                        break
                                else:
                                    st.session_state["messages"].append({"role": "assistant", "content": "❌ Video generation timed out."})
                            else:
                                st.session_state["messages"].append({"role": "assistant", "content": f"❌ Error: {response.status_code} - {response.text}"})
                        except Exception as e:
                            st.session_state["messages"].append({"role": "assistant", "content": f"❌ Video generation error: {e}"})
                    else:
                        st.session_state["messages"].append({"role": "assistant", "content": "❌ Please upload an image to generate a video."})

                threading.Thread(target=task).start()

            def generate_3d_model(prompt_text):
                def task():
                    st.session_state["messages"].append({"role": "assistant", "content": "🔷 Generating 3D model..."})
                    st.experimental_rerun()

                    # Placeholder parameters; in a real scenario, parse from prompt or use defaults
                    data = {
                        "texture_resolution": "1024",
                        "foreground_ratio": 0.85,
                        "remesh": "none",
                        "vertex_count": -1
                    }

                    url = "https://api.stability.ai/v2beta/3d/stable-fast-3d"

                    if uploaded_file and uploaded_file.type.startswith("image/"):
                        try:
                            files = {
                                "image": uploaded_file.read(),
                            }

                            response = requests.post(url, headers=headers_stability, files=files, data=data)
                            if response.status_code == 200:
                                glb_data = response.content
                                filename = f"generated_model_{int(time.time())}.glb"
                                save_file(glb_data, filename)
                                glb_base64 = base64.b64encode(glb_data).decode()
                                model_src = f"data:model/gltf-binary;base64,{glb_base64}"
                                st.session_state["messages"].append({"role": "assistant", "content": "🔷 3D Model generated successfully!", "model_3d": model_src})
                            else:
                                st.session_state["messages"].append({"role": "assistant", "content": f"❌ Error: {response.status_code} - {response.text}"})
                        except Exception as e:
                            st.session_state["messages"].append({"role": "assistant", "content": f"❌ 3D model generation error: {e}"})
                    else:
                        st.session_state["messages"].append({"role": "assistant", "content": "❌ Please upload an image to generate a 3D model."})

                threading.Thread(target=task).start()

            def general_chat(prompt_text):
                def task():
                    st.session_state["messages"].append({"role": "assistant", "content": "💬 Processing your request..."})
                    st.experimental_rerun()

                    chat_payload = {
                        "model": "gpt-4-turbo",
                        "messages": st.session_state["messages"],
                        "max_tokens": 150,
                        "temperature": 0.7
                    }

                    try:
                        response = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers_openai,
                            json=chat_payload
                        )
                        if response.status_code == 200:
                            chat_result = response.json()['choices'][0]['message']['content'].strip()
                            st.session_state["messages"].append({"role": "assistant", "content": chat_result})
                        else:
                            st.session_state["messages"].append({"role": "assistant", "content": f"❌ Error: {response.status_code} - {response.text}"})
                    except Exception as e:
                        st.session_state["messages"].append({"role": "assistant", "content": f"❌ Chat response error: {e}"})

                threading.Thread(target=task).start()

            # Start analysis in a new thread
            threading.Thread(target=analyze_prompt).start()

# --- Tab 2: File Management ---
with tab2:
    st.header("📁 File Management")
    st.subheader("Manage and View Generated Files")

    files = os.listdir(GENERATED_FILES_DIR)
    if files:
        for file in files:
            file_path = os.path.join(GENERATED_FILES_DIR, file)
            st.markdown(f"### {file}")
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                with open(file_path, "rb") as f:
                    img_bytes = f.read()
                    display_image(img_bytes, caption=file)
            elif file.lower().endswith('.mp4'):
                with open(file_path, "rb") as f:
                    video_bytes = f.read()
                    display_video(video_bytes, caption=file)
            elif file.lower().endswith('.glb'):
                with open(file_path, "rb") as f:
                    glb_bytes = f.read()
                    display_3d_model(glb_bytes, caption=file)
            else:
                st.download_button("Download File", data=open(file_path, "rb").read(), file_name=file)
            st.markdown("---")
    else:
        st.info("No files generated yet.")
