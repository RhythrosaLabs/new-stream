import streamlit as st
from openai import OpenAI
import anthropic
from datetime import datetime
import json
import replicate
import stability_sdk
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
if not os.path.exists(FILES_DIR):
    os.makedirs(FILES_DIR)

# Page config and title
st.set_page_config(page_title="🤖 Enhanced AI Assistant Hub", layout="wide")

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

# Function to check temperature support
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
            with st.expander("Document Content"):
                st.text(content)
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")

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
            if uploaded_file and 'content' in locals():
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

# ---------------------- Tab 2: AI Model Comparison ----------------------
with tab2:
    st.header("AI Model Comparison")

    comparison_prompt = st.text_area("Enter text to compare responses across models")

    if st.button("Compare Models"):
        if not (openai_api_key and anthropic_api_key):
            st.info("Please add both OpenAI and Anthropic API keys to compare models.")
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
                    # Save to files
                    image_filename = f"{FILES_DIR}/openai_response_{int(time.time())}.txt"
                    with open(image_filename, "w") as f:
                        f.write(response.choices[0].message.content)

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
                    # Save to files
                    image_filename = f"{FILES_DIR}/anthropic_response_{int(time.time())}.txt"
                    with open(image_filename, "w") as f:
                        f.write(message.content[0].text)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ---------------------- Tab 3: Chat History ----------------------
with tab3:
    st.header("Chat History")

    if st.session_state.conversation_history:
        if st.button("Clear History"):
            st.session_state.conversation_history = []
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your enhanced AI assistant. How can I help you today?"}]
            st.success("History cleared.")

        for record in st.session_state.conversation_history:
            with st.expander(f"Interaction on {record['timestamp']}"):
                st.write("**Model:**", record["model"])
                st.write("**Prompt:**", record["prompt"])
                st.write("**Response:**", record["response"])
    else:
        st.info("No chat history available.")

# ---------------------- Tab 4: Image Generation ----------------------
with tab4:
    st.header("🎨 Image Generation with Stable Diffusion")

    if not stability_api_key:
        st.info("Please enter your Stability AI API key in the sidebar to use this feature.")
        st.stop()

    # Input fields for image generation
    with st.form("image_generation_form"):
        prompt = st.text_area("Enter your image prompt", "A serene landscape with mountains and a river at sunset.")
        negative_prompt = st.text_area("Enter negative prompts (optional)", "low resolution, blurry")
        aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"], index=0)
        seed = st.number_input("Seed (optional)", min_value=0, max_value=4294967294, value=0, step=1)
        output_format = st.selectbox("Output Format", ["png", "jpeg", "webp"], index=0)
        submit_button = st.form_submit_button(label="Generate Image")

    if submit_button:
        if not prompt.strip():
            st.error("Prompt cannot be empty.")
        else:
            try:
                with st.spinner("Generating image..."):
                    # Prepare the API request
                    url = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

                    headers = {
                        "Authorization": f"Bearer {stability_api_key}",
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

                        st.image(image, caption="Generated Image", use_column_width=True)

                        # Provide download option
                        buffered = io.BytesIO()
                        image.save(buffered, format=output_format.upper())
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        href = f'<a href="data:image/{output_format};base64,{img_base64}" download="{os.path.basename(image_filename)}">Download Image</a>'
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

    if not stability_api_key:
        st.info("Please enter your Stability AI API key in the sidebar to use this feature.")
        st.stop()

    generation_id = st.session_state.get("video_generation_id", None)

    # Input fields for video generation
    with st.form("video_generation_form"):
        image = st.file_uploader("Upload an initial image", type=["png", "jpeg", "jpg"])
        cfg_scale = st.slider("CFG Scale", 0.0, 10.0, 1.8)
        motion_bucket_id = st.slider("Motion Bucket ID", 1, 255, 127)
        seed = st.number_input("Seed (optional)", min_value=0, max_value=4294967294, value=0, step=1)
        submit_button = st.form_submit_button(label="Start Video Generation")

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
                        "Authorization": f"Bearer {stability_api_key}"
                    }

                    data = {
                        "cfg_scale": cfg_scale,
                        "motion_bucket_id": motion_bucket_id
                    }

                    if seed != 0:
                        data["seed"] = seed

                    # Save the uploaded image temporarily
                    temp_image_path = f"{FILES_DIR}/temp_image_{int(time.time())}.{image.type.replace('image/', '')}"
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
        st.info(f"Polling for video generation result. ID: {generation_id}")
        with st.spinner("Checking video generation status..."):
            try:
                poll_url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"
                headers = {
                    "Authorization": f"Bearer {stability_api_key}",
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
                        href = f'<a href="data:video/mp4;base64,{video_base64}" download="{os.path.basename(video_filename)}">Download Video</a>'
                        st.markdown(href, unsafe_allow_html=True)

                    st.success("Video generation completed successfully!")

                    # Optionally, remove the video after download
                    # os.remove(video_filename)

                    # Clear the generation ID
                    st.session_state.video_generation_id = None
                elif response.status_code == 202:
                    st.warning("Video generation is still in-progress. Please wait and try again later.")
                else:
                    try:
                        error_info = response.json()
                        error_message = error_info.get("detail", "An error occurred.")
                    except ValueError:
                        error_message = response.text
                    st.error(f"Error {response.status_code}: {error_message}")
            except Exception as e:
                st.error(f"An error occurred while fetching the video: {str(e)}")

# ---------------------- Tab 6: Music Generation ----------------------
with tab6:
    st.header("🎵 Music Generation with Replicate's MusicGen")

    if not replicate_api_key:
        st.info("Please enter your Replicate API key in the sidebar to use this feature.")
        st.stop()

    # Initialize Replicate client
    replicate_client = replicate.Client(api_token=replicate_api_key)

    # Input fields for music generation
    with st.form("music_generation_form"):
        prompt = st.text_area("Enter your music prompt", "A triumphant and cinematic orchestral piece with a 9th harmonic resolution.")
        version_id = st.selectbox("Select MusicGen Model Version", [
            "stereo-large",  # Replace with actual Replicate MusicGen version IDs
            "mono-medium",
            "stereo-small"
        ], index=0)
        submit_button = st.form_submit_button(label="Generate Music")

    if submit_button:
        if not prompt.strip():
            st.error("Prompt cannot be empty.")
        else:
            try:
                with st.spinner("Generating music..."):
                    # Create prediction
                    prediction = replicate_client.predictions.create(
                        model="meta/musicgen:latest",  # Replace with actual model identifier
                        version=version_id,
                        input={"prompt": prompt}
                    )

                    # Poll for prediction status
                    st.info("Polling for music generation result...")
                    while True:
                        prediction = replicate_client.predictions.get(prediction.id)
                        if prediction.status == "succeeded":
                            break
                        elif prediction.status == "failed":
                            st.error("Music generation failed.")
                            st.stop()
                        time.sleep(10)  # Poll every 10 seconds

                    # Download the generated music
                    output_url = prediction.output
                    response = requests.get(output_url)

                    if response.status_code == 200:
                        audio_data = response.content
                        audio_filename = f"{FILES_DIR}/music_{int(time.time())}.mp3"
                        with open(audio_filename, "wb") as f:
                            f.write(audio_data)

                        st.audio(audio_filename, format="audio/mp3")

                        # Provide download option
                        with open(audio_filename, "rb") as f:
                            audio_bytes = f.read()
                            audio_base64 = base64.b64encode(audio_bytes).decode()
                            href = f'<a href="data:audio/mp3;base64,{audio_base64}" download="{os.path.basename(audio_filename)}">Download Music</a>'
                            st.markdown(href, unsafe_allow_html=True)

                        st.success("Music generation completed successfully!")
                    else:
                        st.error("Failed to download the generated music.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# ---------------------- Tab 7: Files ----------------------
with st.tabs(["🎨 Image Generation", "🎥 Image-to-Video", "🎵 Music Generation"]):
    with tab4:
        pass  # Existing Image Generation tab
    with tab5:
        pass  # Existing Image-to-Video tab
    with tab6:
        pass  # Existing Music Generation tab

with st.expander("View and Manage Generated Files"):
    generated_files = os.listdir(FILES_DIR)
    if generated_files:
        for file in generated_files:
            file_path = os.path.join(FILES_DIR, file)
            file_extension = file.split('.')[-1].lower()
            if file_extension in ["png", "jpeg", "jpg", "webp"]:
                st.image(file_path, caption=file, use_column_width=True)
            elif file_extension in ["mp4", "avi", "mov"]:
                st.video(file_path)
            elif file_extension in ["mp3", "wav", "ogg"]:
                st.audio(file_path, format=f"audio/{file_extension}")
            else:
                st.write(f"**{file}**")

            # Provide download option
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                file_base64 = base64.b64encode(file_bytes).decode()
                if file_extension in ["png", "jpeg", "jpg", "webp"]:
                    mime = f"image/{file_extension}"
                elif file_extension in ["mp4", "avi", "mov"]:
                    mime = "video/mp4"
                elif file_extension in ["mp3", "wav", "ogg"]:
                    mime = "audio/mp3"
                else:
                    mime = "application/octet-stream"
                href = f'<a href="data:{mime};base64,{file_base64}" download="{file}">Download {file}</a>'
                st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("No generated files available.")

# ---------------------- Optional: Clear Files ----------------------
with st.expander("Manage Files"):
    if st.button("Clear All Files"):
        try:
            shutil.rmtree(FILES_DIR)
            os.makedirs(FILES_DIR)
            st.success("All generated files have been cleared.")
        except Exception as e:
            st.error(f"An error occurred while clearing files: {str(e)}")
