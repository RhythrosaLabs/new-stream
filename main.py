import streamlit as st
from openai import OpenAI
import anthropic
import replicate
import stability_sdk
from datetime import datetime
import json
import requests
import base64
from PIL import Image
import io
import os
import time
import shutil

# Constants
FILES_DIR = "generated_files"

# Ensure the files directory exists
os.makedirs(FILES_DIR, exist_ok=True)

# Page configuration and title
st.set_page_config(
    page_title="🤖 Enhanced AI Assistant Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for improved UI
st.markdown("""
<style>
    /* Adjust the spacing and styling of tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        white-space: pre-wrap; 
        background-color: #f0f2f6; 
        border-radius: 5px; 
        gap: 12px; 
        padding: 10px 16px; 
    }
    .stTabs [aria-selected="true"] { background-color: #e0e2e6; }
    /* Enhance the main header */
    .main-header {
        font-size: 2.5em;
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Style buttons uniformly */
    .stButton > button { width: 100%; }
    /* Add spacing to input fields */
    .stTextInput, .stNumberInput, .stSelectbox, .stRadio, .stSlider { margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 Enhanced AI Assistant Hub</h1>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("🔑 API Configuration")

    # API Key inputs with session state handling
    api_keys = {
        "openai_api_key": st.text_input("OpenAI API Key", key="openai_api_key", type="password"),
        "anthropic_api_key": st.text_input("Anthropic API Key", key="anthropic_api_key", type="password"),
        "replicate_api_key": st.text_input("Replicate API Key", key="replicate_api_key", type="password"),
        "stability_api_key": st.text_input("Stability API Key", key="stability_api_key", type="password"),
    }

    # Save/Load API Keys
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save API Keys"):
            with open("api_keys.json", "w") as f:
                json.dump(api_keys, f)
            st.success("API Keys saved successfully!")
    with col2:
        if st.button("📂 Load API Keys"):
            try:
                with open("api_keys.json", "r") as f:
                    saved_keys = json.load(f)
                for key in api_keys:
                    if key in saved_keys:
                        api_keys[key] = saved_keys[key]
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
    - [Stability AI API Keys](https://platform.stability.ai/)
    - [Replicate API Keys](https://replicate.com/account/apikey)
    """)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your enhanced AI assistant. How can I help you today?"}]
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "video_generation_id" not in st.session_state:
    st.session_state.video_generation_id = None

# Function to check if the model supports temperature
def model_supports_temperature(model_name):
    unsupported_models = {"o1-preview", "o1-mini"}
    return model_name not in unsupported_models

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Chat & Document Analysis", 
    "🔄 AI Model Comparison", 
    "📊 Chat History", 
    "🎨 Image Generation", 
    "🎥 Image-to-Video", 
    "🎵 Music Generation"
])

# ---------------------- Tab 1: Chat & Document Analysis ----------------------
with tab1:
    st.header("💬 Chat & Document Analysis")

    # Model selector
    chat_model = st.radio("Select AI Model", ["OpenAI GPT", "Anthropic Claude"], horizontal=True)

    # Chat display
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
        else:
            st.chat_message("user").write(msg["content"])

    # File upload and display content
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf", "docx"])
    if uploaded_file:
        st.session_state.current_file = uploaded_file
        try:
            if uploaded_file.type == "application/pdf":
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                pages = [page.extract_text() for page in pdf_reader.pages]
                content = "\n".join(pages)
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                import docx
                doc = docx.Document(uploaded_file)
                content = "\n".join([para.text for para in doc.paragraphs])
            else:
                content = uploaded_file.read().decode()
            with st.expander("📄 Document Content"):
                st.text(content)
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")

    # Chat input
    if chat_prompt := st.chat_input("Send a message or ask a question about the document"):
        if chat_model == "OpenAI GPT" and not api_keys["openai_api_key"]:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        elif chat_model == "Anthropic Claude" and not api_keys["anthropic_api_key"]:
            st.info("Please add your Anthropic API key to continue.")
            st.stop()

        # Send message
        try:
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            st.chat_message("user").write(chat_prompt)

            # Set up question based on chat or document
            question = chat_prompt
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    question = f"Here's a PDF document:\n\n{content}\n\n{chat_prompt}"
                elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    question = f"Here's a DOCX document:\n\n{content}\n\n{chat_prompt}"
                else:
                    question = f"Here's a document:\n\n{content}\n\n{chat_prompt}"

            if chat_model == "OpenAI GPT":
                client = OpenAI(api_key=api_keys["openai_api_key"])

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
                client = anthropic.Client(api_key=api_keys["anthropic_api_key"])
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

# ---------------------- Tab 2: AI Model Comparison ----------------------
with tab2:
    st.header("🔄 AI Model Comparison")

    comparison_prompt = st.text_area("Enter text to compare responses across models")

    if st.button("Compare Models"):
        if not (api_keys["openai_api_key"] and api_keys["anthropic_api_key"]):
            st.info("Please add both OpenAI and Anthropic API keys to compare models.")
            st.stop()

        try:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("OpenAI GPT Response")
                with st.spinner("Generating response..."):
                    client = OpenAI(api_key=api_keys["openai_api_key"])
                    response = client.chat.completions.create(
                        model=openai_model,
                        messages=[{"role": "user", "content": comparison_prompt}],
                        temperature=temperature
                    )
                    openai_response = response.choices[0].message.content
                    st.write(openai_response)
                    # Save to files
                    response_filename = f"{FILES_DIR}/openai_response_{int(time.time())}.txt"
                    with open(response_filename, "w") as f:
                        f.write(openai_response)

            with col2:
                st.subheader("Anthropic Claude Response")
                with st.spinner("Generating response..."):
                    client = anthropic.Client(api_key=api_keys["anthropic_api_key"])
                    message = client.messages.create(
                        model=anthropic_model,
                        max_tokens=2000,
                        temperature=temperature,
                        messages=[{"role": "user", "content": comparison_prompt}]
                    )
                    anthropic_response = message.content[0].text
                    st.write(anthropic_response)
                    # Save to files
                    response_filename = f"{FILES_DIR}/anthropic_response_{int(time.time())}.txt"
                    with open(response_filename, "w") as f:
                        f.write(anthropic_response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ---------------------- Tab 3: Chat History ----------------------
with tab3:
    st.header("📊 Chat History")

    if st.session_state.conversation_history:
        col1, col2 = st.columns([3, 1])
        with col1:
            for record in reversed(st.session_state.conversation_history):
                with st.expander(f"🕒 {record['timestamp']} - **{record['model']}**"):
                    st.markdown(f"**You:** {record['prompt']}")
                    st.markdown(f"**Assistant:** {record['response']}")
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.conversation_history = []
                st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your enhanced AI assistant. How can I help you today?"}]
                st.success("History cleared.")
    else:
        st.info("No chat history available.")

# ---------------------- Tab 4: Image Generation ----------------------
with tab4:
    st.header("🎨 Image Generation with Stable Diffusion")

    if not api_keys["stability_api_key"]:
        st.info("Please enter your Stability AI API key in the sidebar to use this feature.")
        st.stop()

    # Input fields for image generation
    with st.form("image_generation_form"):
        prompt = st.text_area("Enter your image prompt", "A serene landscape with mountains and a river at sunset.")
        negative_prompt = st.text_area("Enter negative prompts (optional)", "low resolution, blurry")
        aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"], index=0)
        seed = st.number_input("Seed (optional)", min_value=0, max_value=4294967294, value=0, step=1)
        output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)
        submit_button = st.form_submit_button(label="🎯 Generate Image")

    if submit_button:
        if not prompt.strip():
            st.error("Prompt cannot be empty.")
        else:
            try:
                with st.spinner("Generating image..."):
                    # Prepare the API request
                    url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

                    headers = {
                        "Authorization": f"Bearer {api_keys['stability_api_key']}",
                        "Accept": "image/*"
                    }

                    data = {
                        "prompt": prompt,
                        "output_format": output_format,
                        "aspect_ratio": aspect_ratio
                    }

                    if negative_prompt.strip():
                        data["negative_prompt"] = negative_prompt

                    if seed != 0:
                        data["seed"] = seed

                    files = {
                        "file": ("", "")  # Dummy file field as per API requirement
                    }

                    response = requests.post(url, headers=headers, data=data, files=files)

                    if response.status_code == 200:
                        image_data = response.content
                        image = Image.open(io.BytesIO(image_data))

                        # Save image to files directory
                        image_filename = f"{FILES_DIR}/image_{int(time.time())}.{output_format}"
                        image.save(image_filename)

                        st.image(image, caption="🖼️ Generated Image", use_column_width=True)

                        # Provide download option
                        buffered = io.BytesIO()
                        image.save(buffered, format=output_format.upper())
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        href = f'<a href="data:image/{output_format};base64,{img_base64}" download="{os.path.basename(image_filename)}">⬇️ Download Image</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        st.success("Image generated successfully!")
                    elif response.status_code == 429:
                        st.error("Rate limit exceeded. Please wait and try again later.")
                    else:
                        try:
                            error_info = response.json()
                            error_message = error_info.get("detail", "An error occurred.")
                        except ValueError:
                            error_message = response.text
                        st.error(f"Error {response.status_code}: {error_message}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# ---------------------- Tab 5: Image-to-Video ----------------------
with tab5:
    st.header("🎥 Image-to-Video with Stable Video Diffusion")

    if not api_keys["stability_api_key"]:
        st.info("Please enter your Stability AI API key in the sidebar to use this feature.")
        st.stop()

    generation_id = st.session_state.get("video_generation_id", None)

    # Input fields for video generation
    with st.form("video_generation_form"):
        image = st.file_uploader("Upload an initial image", type=["png", "jpeg", "jpg"])
        cfg_scale = st.slider("CFG Scale", 0.0, 10.0, 1.8)
        motion_bucket_id = st.slider("Motion Bucket ID", 1, 255, 127)
        seed = st.number_input("Seed (optional)", min_value=0, max_value=4294967294, value=0, step=1)
        submit_button = st.form_submit_button(label="▶️ Start Video Generation")

    if submit_button:
        if not image:
            st.error("Please upload an image to generate a video.")
        elif image.size > 10 * 1024 * 1024:
            st.error("The uploaded image exceeds the size limit of 10MiB.")
        else:
            try:
                with st.spinner("Starting video generation..."):
                    url = "https://api.stability.ai/v2beta/image-to-video"

                    headers = {
                        "Authorization": f"Bearer {api_keys['stability_api_key']}"
                    }

                    data = {
                        "cfg_scale": cfg_scale,
                        "motion_bucket_id": motion_bucket_id
                    }

                    if seed != 0:
                        data["seed"] = seed

                    # Save the uploaded image temporarily
                    image_extension = image.type.replace('image/', '')
                    temp_image_path = f"{FILES_DIR}/temp_image_{int(time.time())}.{image_extension}"
                    with open(temp_image_path, "wb") as f:
                        f.write(image.getbuffer())

                    files = {
                        "image": open(temp_image_path, "rb")
                    }

                    response = requests.post(url, headers=headers, data=data, files=files)

                    # Remove the temporary image
                    os.remove(temp_image_path)

                    if response.status_code == 200:
                        generation_id = response.json().get('id')
                        st.success(f"Video generation started. ID: {generation_id}")
                        st.session_state.video_generation_id = generation_id
                    elif response.status_code == 429:
                        st.error("Rate limit exceeded. Please wait and try again later.")
                    else:
                        try:
                            error_info = response.json()
                            error_message = error_info.get("detail", "An error occurred.")
                        except ValueError:
                            error_message = response.text
                        st.error(f"Error {response.status_code}: {error_message}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    if generation_id:
        st.info(f"🔄 Polling for video generation result. ID: {generation_id}")
        with st.spinner("Checking video generation status..."):
            try:
                poll_url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"
                headers = {
                    "Authorization": f"Bearer {api_keys['stability_api_key']}",
                    "Accept": "video/*"
                }
                response = requests.get(poll_url, headers=headers)

                if response.status_code == 200:
                    video_data = response.content
                    video_filename = f"{FILES_DIR}/video_{int(time.time())}.mp4"
                    with open(video_filename, "wb") as f:
                        f.write(video_data)

                    st.video(video_filename)

                    # Provide download option
                    with open(video_filename, "rb") as f:
                        video_bytes = f.read()
                        video_base64 = base64.b64encode(video_bytes).decode()
                        href = f'<a href="data:video/mp4;base64,{video_base64}" download="{os.path.basename(video_filename)}">⬇️ Download Video</a>'
                        st.markdown(href, unsafe_allow_html=True)

                    st.success("Video generated successfully!")
                    # Reset generation ID after successful download
                    st.session_state.video_generation_id = None
                elif response.status_code == 202:
                    st.info("Video is still being generated. Please wait...")
                elif response.status_code == 429:
                    st.error("Rate limit exceeded. Please wait and try again later.")
                else:
                    try:
                        error_info = response.json()
                        error_message = error_info.get("detail", "An error occurred.")
                    except ValueError:
                        error_message = response.text
                    st.error(f"Error {response.status_code}: {error_message}")
            except Exception as e:
                st.error(f"An error occurred while polling: {str(e)}")

# ---------------------- Tab 6: Music Generation ----------------------
with tab6:
    st.header("🎵 Music Generation")

    if not api_keys["replicate_api_key"]:
        st.info("Please enter your Replicate API key in the sidebar to use this feature.")
        st.stop()

    # Input fields for music generation
    with st.form("music_generation_form"):
        music_prompt = st.text_area("Enter your music description", "A relaxing instrumental with piano and strings.")
        duration = st.number_input("Duration (seconds)", min_value=10, max_value=600, value=60, step=10)
        genre = st.selectbox("Genre", ["Classical", "Jazz", "Pop", "Electronic", "Ambient", "Rock", "Hip Hop"])
        tempo = st.slider("Tempo (BPM)", min_value=60, max_value=180, value=120)
        submit_button = st.form_submit_button(label="🎶 Generate Music")

    if submit_button:
        if not music_prompt.strip():
            st.error("Music description cannot be empty.")
        else:
            try:
                with st.spinner("Generating music..."):
                    # Example using Replicate's music generation model
                    model = replicate.models.get("your-model/music-generator")
                    output = model.predict(
                        prompt=music_prompt,
                        duration=duration,
                        genre=genre,
                        tempo=tempo
                    )
                    
                    # Assuming the model returns a URL to the generated music file
                    music_url = output.get("audio_url")
                    if music_url:
                        # Download the music file
                        music_response = requests.get(music_url)
                        if music_response.status_code == 200:
                            music_filename = f"{FILES_DIR}/music_{int(time.time())}.mp3"
                            with open(music_filename, "wb") as f:
                                f.write(music_response.content)
                            
                            st.audio(music_filename, format='audio/mp3')
                            
                            # Provide download option
                            with open(music_filename, "rb") as f:
                                music_bytes = f.read()
                                music_base64 = base64.b64encode(music_bytes).decode()
                                href = f'<a href="data:audio/mp3;base64,{music_base64}" download="{os.path.basename(music_filename)}">⬇️ Download Music</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            
                            st.success("Music generated successfully!")
                        else:
                            st.error("Failed to download the generated music.")
                    else:
                        st.error("Music generation failed. Please try again.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
