import streamlit as st
import os
import requests
import threading
import time
import base64
from io import BytesIO
from PIL import Image
import json

# Directory to save generated files
GENERATED_FILES_DIR = "generated_files"
if not os.path.exists(GENERATED_FILES_DIR):
    os.makedirs(GENERATED_FILES_DIR)

# Custom CSS for layout
st.markdown("""
    <style>
        .tab-container {
            padding: 20px;
        }
        .section-title {
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        .generated-image, .generated-video, .generated-model {
            max-width: 100%;
            height: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for API Key Configuration
st.sidebar.header("🔑 API Configuration")
st.sidebar.markdown("Enter your Stability AI API Key below:")
api_key = st.sidebar.text_input("Stability AI API Key", type="password")

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

# Tabs for different functionalities
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔑 API Configuration",
    "🖼️ Image Generation & Editing",
    "🎞️ Video Generation",
    "🔷 3D Generation",
    "📁 File Management"
])

# --- Tab 1: API Configuration ---
with tab1:
    st.header("🔑 API Configuration")
    st.markdown("Enter your Stability AI API Key to enable all features.")
    if api_key:
        st.success("API Key is set and verified.")
        st.session_state["api_key"] = api_key
    else:
        st.warning("Please enter your Stability AI API Key in the sidebar.")

# Ensure API Key is provided for other tabs
if 'api_key' not in st.session_state and tab2 != st.session_state.get('current_tab'):
    st.warning("Please enter your Stability AI API Key in the sidebar to access other features.")

# --- Tab 2: Image Generation & Editing ---
with tab2:
    st.header("🖼️ Image Generation & Editing")
    if 'api_key' not in st.session_state:
        st.warning("Please enter your Stability AI API Key in the sidebar to access this feature.")
    else:
        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            "📝 Text-to-Image",
            "🖼️ Image-to-Image",
            "✨ Image Effects",
            "🎨 Canvas"
        ])

        # --- Sub-Tab 2.1: Text-to-Image ---
        with sub_tab1:
            st.subheader("📝 Text-to-Image")
            with st.form(key='text_to_image_form'):
                model_type = st.selectbox("Select Model:", [
                    "Stable Image Ultra", "Stable Image Core",
                    "Stable Diffusion 3.5 Large", "Stable Diffusion 3.5 Large Turbo",
                    "Stable Diffusion 3.0 Large", "Stable Diffusion 3.0 Large Turbo", "Stable Diffusion 3.0 Medium"
                ])
                prompt = st.text_area("Prompt:")
                negative_prompt = st.text_area("Negative Prompt:")
                aspect_ratio = st.selectbox("Aspect Ratio:", ["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"])
                seed = st.number_input("Seed (0 for random):", min_value=0, step=1, value=0)
                output_format = st.selectbox("Output Format:", ["png", "jpeg", "webp"])
                cfg_scale = st.slider("CFG Scale:", 0.0, 35.0, 7.0, 0.1)
                steps = st.slider("Steps:", 1, 150, 50)
                sampler = st.selectbox("Sampler:", [
                    "DDIM", "DDPM", "K_DPMPP_2M", "K_DPMPP_2S_ANCESTRAL",
                    "K_DPM_2", "K_DPM_2_ANCESTRAL", "K_EULER",
                    "K_EULER_ANCESTRAL", "K_HEUN", "K_LMS"
                ])
                samples = st.number_input("Samples:", min_value=1, max_value=10, value=1)
                submit_tti = st.form_submit_button("Generate Image")

            if submit_tti:
                def generate_text_to_image():
                    st.info("Generating image...")
                    data = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "aspect_ratio": aspect_ratio,
                        "seed": seed,
                        "output_format": output_format,
                        "cfg_scale": cfg_scale,
                        "steps": steps,
                        "sampler": sampler,
                        "samples": samples,
                        "model": model_type.lower().replace(" ", "-")
                    }
                    headers = {
                        "Authorization": f"Bearer {st.session_state['api_key']}",
                        "Accept": "application/json"
                    }
                    url = "https://api.stability.ai/v2beta/stable-image/generate"

                    response = requests.post(url, headers=headers, data=data)
                    if response.status_code == 200:
                        data = response.json()
                        if 'artifacts' in data:
                            img_data = base64.b64decode(data['artifacts'][0]['base64'])
                            filename = f"text_to_image_{int(time.time())}.{output_format}"
                            save_file(img_data, filename)
                            st.success("Image generated successfully!")
                            display_image(img_data, caption=prompt)
                        else:
                            st.error("No artifacts found in the response.")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")

                generate_text_to_image()

        # --- Sub-Tab 2.2: Image-to-Image ---
        with sub_tab2:
            st.subheader("🖼️ Image-to-Image")
            uploaded_img = st.file_uploader("Upload Image for Modification", type=["png", "jpg", "jpeg", "bmp"])

            if uploaded_img:
                image = Image.open(uploaded_img)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                with st.form(key='image_to_image_form'):
                    model_type = st.selectbox("Select Model:", [
                        "Stable Diffusion 3.5 Large", "Stable Diffusion 3.5 Large Turbo",
                        "Stable Diffusion 3.0 Large", "Stable Diffusion 3.0 Large Turbo", "Stable Diffusion 3.0 Medium"
                    ])
                    prompt = st.text_area("Prompt:")
                    negative_prompt = st.text_area("Negative Prompt:")
                    image_strength = st.slider("Image Strength:", 0.0, 1.0, 0.5, 0.01)
                    seed = st.number_input("Seed (0 for random):", min_value=0, step=1, value=0)
                    output_format = st.selectbox("Output Format:", ["png", "jpeg", "webp"])
                    steps = st.slider("Steps:", 1, 150, 50)
                    sampler = st.selectbox("Sampler:", [
                        "DDIM", "DDPM", "K_DPMPP_2M", "K_DPMPP_2S_ANCESTRAL",
                        "K_DPM_2", "K_DPM_2_ANCESTRAL", "K_EULER",
                        "K_EULER_ANCESTRAL", "K_HEUN", "K_LMS"
                    ])
                    cfg_scale = st.slider("CFG Scale:", 0.0, 35.0, 7.0, 0.1)
                    samples = st.number_input("Samples:", min_value=1, max_value=10, value=1)
                    submit_iti = st.form_submit_button("Generate Image")

                if submit_iti:
                    def generate_image_to_image():
                        st.info("Generating modified image...")
                        data = {
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "seed": seed,
                            "output_format": output_format,
                            "strength": image_strength,
                            "mode": "image-to-image",
                            "model": model_type.lower().replace(" ", "-"),
                            "steps": steps,
                            "sampler": sampler,
                            "cfg_scale": cfg_scale,
                            "samples": samples,
                        }
                        headers = {
                            "Authorization": f"Bearer {st.session_state['api_key']}",
                            "Accept": "application/json"
                        }
                        url = "https://api.stability.ai/v2beta/stable-image/generate"

                        files = {
                            "image": uploaded_img.read(),
                        }

                        response = requests.post(url, headers=headers, files=files, data=data)
                        if response.status_code == 200:
                            data = response.json()
                            if 'artifacts' in data:
                                img_data = base64.b64decode(data['artifacts'][0]['base64'])
                                filename = f"image_to_image_{int(time.time())}.{output_format}"
                                save_file(img_data, filename)
                                st.success("Image generated successfully!")
                                display_image(img_data, caption=prompt)
                            else:
                                st.error("No artifacts found in the response.")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")

                    generate_image_to_image()

        # --- Sub-Tab 2.3: Image Effects ---
        with sub_tab3:
            st.subheader("✨ Image Effects")
            uploaded_effect_img = st.file_uploader("Upload Image for Effects", type=["png", "jpg", "jpeg", "bmp"])

            if uploaded_effect_img:
                image = Image.open(uploaded_effect_img)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                effect = st.selectbox("Select Effect:", ["Upscale", "Remove Background"])

                with st.form(key='image_effects_form'):
                    if effect == "Upscale":
                        upscale_type = st.selectbox("Upscale Type:", ["Fast", "Conservative", "Creative"])
                        output_format = st.selectbox("Output Format:", ["png", "jpeg", "webp"])
                        prompt_upscale = st.text_area("Upscale Prompt:")
                        negative_prompt_upscale = st.text_area("Negative Prompt:")
                        seed_upscale = st.number_input("Seed (0 for random):", min_value=0, step=1, value=0)
                        creativity = st.slider("Creativity:", 0.0, 1.0, 0.35, 0.01)
                    elif effect == "Remove Background":
                        output_format_remove_bg = st.selectbox("Output Format:", ["png", "jpeg", "webp"])
                    submit_effect = st.form_submit_button("Apply Effect")

                if submit_effect:
                    def apply_image_effect():
                        st.info("Applying effect...")
                        headers = {
                            "Authorization": f"Bearer {st.session_state['api_key']}",
                            "Accept": "application/json"
                        }
                        files = {
                            "image": uploaded_effect_img.read(),
                        }
                        data = {}
                        url = ""
                        if effect == "Upscale":
                            data["output_format"] = output_format
                            if upscale_type == "Fast":
                                url = "https://api.stability.ai/v2beta/stable-image/upscale/fast"
                            else:
                                data["prompt"] = prompt_upscale
                                data["negative_prompt"] = negative_prompt_upscale
                                data["seed"] = seed_upscale
                                data["creativity"] = creativity
                                if upscale_type == "Conservative":
                                    url = "https://api.stability.ai/v2beta/stable-image/upscale/conservative"
                                else:
                                    url = "https://api.stability.ai/v2beta/stable-image/upscale/creative"
                        elif effect == "Remove Background":
                            data["output_format"] = output_format_remove_bg
                            url = "https://api.stability.ai/v2beta/stable-image/edit/remove-background"

                        response = requests.post(url, headers=headers, files=files, data=data)
                        if response.status_code == 200:
                            img_data = response.content
                            filename = f"{effect.lower()}_{int(time.time())}.{output_format if effect == 'Upscale' else output_format_remove_bg}"
                            save_file(img_data, filename)
                            st.success("Effect applied successfully!")
                            if effect == "Upscale":
                                display_image(img_data, caption="Upscaled Image")
                            elif effect == "Remove Background":
                                display_image(img_data, caption="Background Removed Image")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")

                    apply_image_effect()

        # --- Sub-Tab 2.4: Canvas ---
        with sub_tab4:
            st.subheader("🎨 Canvas")
            st.write("Draw on the canvas or upload an image to modify.")

            canvas_mode = st.selectbox("Canvas Mode:", ["Draw", "Upload Image"])

            if canvas_mode == "Draw":
                import streamlit_drawable_canvas as st_canvas

                stroke_width = st.slider("Stroke Width:", 1, 25, 3)
                stroke_color = st.color_picker("Stroke Color:", "#000000")
                bg_color = st.color_picker("Background Color:", "#FFFFFF")
                drawing_mode = st.selectbox("Drawing Mode:", ["freedraw", "transform"])

                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    update_streamlit=True,
                    height=512,
                    width=512,
                    drawing_mode=drawing_mode,
                    key="canvas",
                )

                if canvas_result.image_data is not None:
                    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.image(img, caption="Canvas Image", use_column_width=True)

                    with st.form(key='canvas_image_form'):
                        prompt_canvas = st.text_area("Prompt for Canvas Image Modification:")
                        submit_canvas = st.form_submit_button("Generate Image from Canvas")

                    if submit_canvas:
                        def generate_from_canvas():
                            st.info("Generating image from canvas...")
                            data = {
                                "prompt": prompt_canvas,
                                "model": "stable-diffusion-3.0-large",
                                "steps": 50,
                                "sampler": "DDIM",
                                "cfg_scale": 7.0,
                                "seed": 0,
                                "output_format": "png",
                                "mode": "image-to-image",
                                "strength": 0.5
                            }
                            headers = {
                                "Authorization": f"Bearer {st.session_state['api_key']}",
                                "Accept": "application/json"
                            }
                            url = "https://api.stability.ai/v2beta/stable-image/generate"

                            files = {
                                "image": buf.getvalue(),
                            }

                            response = requests.post(url, headers=headers, files=files, data=data)
                            if response.status_code == 200:
                                data = response.json()
                                if 'artifacts' in data:
                                    img_data = base64.b64decode(data['artifacts'][0]['base64'])
                                    filename = f"canvas_image_{int(time.time())}.png"
                                    save_file(img_data, filename)
                                    st.success("Image generated successfully from canvas!")
                                    display_image(img_data, caption=prompt_canvas)
                                else:
                                    st.error("No artifacts found in the response.")
                            else:
                                st.error(f"Error: {response.status_code} - {response.text}")

                        generate_from_canvas()

            elif canvas_mode == "Upload Image":
                uploaded_canvas_img = st.file_uploader("Upload Image for Canvas", type=["png", "jpg", "jpeg", "bmp"])
                if uploaded_canvas_img:
                    image = Image.open(uploaded_canvas_img)
                    st.image(image, caption="Uploaded Image", use_column_width=True)

                    with st.form(key='upload_canvas_form'):
                        prompt_upload = st.text_area("Prompt for Image Modification:")
                        submit_upload = st.form_submit_button("Generate Image from Uploaded Image")

                    if submit_upload:
                        def generate_from_uploaded_image():
                            st.info("Generating image from uploaded image...")
                            data = {
                                "prompt": prompt_upload,
                                "model": "stable-diffusion-3.0-large",
                                "steps": 50,
                                "sampler": "DDIM",
                                "cfg_scale": 7.0,
                                "seed": 0,
                                "output_format": "png",
                                "mode": "image-to-image",
                                "strength": 0.5
                            }
                            headers = {
                                "Authorization": f"Bearer {st.session_state['api_key']}",
                                "Accept": "application/json"
                            }
                            url = "https://api.stability.ai/v2beta/stable-image/generate"

                            files = {
                                "image": uploaded_canvas_img.read(),
                            }

                            response = requests.post(url, headers=headers, files=files, data=data)
                            if response.status_code == 200:
                                data = response.json()
                                if 'artifacts' in data:
                                    img_data = base64.b64decode(data['artifacts'][0]['base64'])
                                    filename = f"uploaded_canvas_image_{int(time.time())}.png"
                                    save_file(img_data, filename)
                                    st.success("Image generated successfully from uploaded image!")
                                    display_image(img_data, caption=prompt_upload)
                                else:
                                    st.error("No artifacts found in the response.")
                            else:
                                st.error(f"Error: {response.status_code} - {response.text}")

                        generate_from_uploaded_image()

# --- Tab 3: Video Generation ---
with tab3:
    st.header("🎞️ Video Generation")
    if 'api_key' not in st.session_state:
        st.warning("Please enter your Stability AI API Key in the sidebar to access this feature.")
    else:
        st.subheader("Generate Video from Image")
        uploaded_video_img = st.file_uploader("Upload Initial Image for Video Generation", type=["png", "jpg", "jpeg", "bmp"])

        if uploaded_video_img:
            image = Image.open(uploaded_video_img)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.form(key='video_generation_form'):
                cfg_scale_video = st.slider("CFG Scale:", 0.0, 10.0, 1.8, 0.1)
                motion_bucket_id = st.number_input("Motion Bucket ID:", min_value=1, max_value=255, value=127, step=1)
                seed_video = st.number_input("Seed (0 for random):", min_value=0, step=1, value=0)
                submit_video = st.form_submit_button("Generate Video")

            if submit_video:
                def generate_video():
                    st.info("Generating video...")
                    data = {
                        "cfg_scale": cfg_scale_video,
                        "motion_bucket_id": motion_bucket_id,
                        "seed": seed_video
                    }
                    headers = {
                        "Authorization": f"Bearer {st.session_state['api_key']}",
                    }
                    url = "https://api.stability.ai/v2beta/image-to-video"

                    files = {
                        "image": uploaded_video_img.read(),
                    }

                    response = requests.post(url, headers=headers, files=files, data=data)
                    if response.status_code == 200:
                        generation_id = response.json().get("id")
                        st.info("Video generation started. Polling for result...")

                        result_url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"
                        accept_header = "video/*"
                        max_retries = 30
                        retry_delay = 10  # seconds

                        for attempt in range(max_retries):
                            time.sleep(retry_delay)
                            result_response = requests.get(
                                result_url,
                                headers={
                                    "Authorization": f"Bearer {st.session_state['api_key']}",
                                    "Accept": accept_header,
                                },
                            )
                            if result_response.status_code == 200:
                                video_bytes = result_response.content
                                filename = f"generated_video_{int(time.time())}.mp4"
                                save_file(video_bytes, filename)
                                st.success("Video generated successfully!")
                                display_video(video_bytes, caption="Generated Video")
                                break
                            elif result_response.status_code == 202:
                                st.info(f"Still processing... ({attempt + 1}/{max_retries})")
                                continue
                            else:
                                st.error(f"Error: {result_response.status_code} - {result_response.text}")
                                break
                        else:
                            st.error("Video generation timed out.")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")

                generate_video()

# --- Tab 4: 3D Generation ---
with tab4:
    st.header("🔷 3D Generation")
    if 'api_key' not in st.session_state:
        st.warning("Please enter your Stability AI API Key in the sidebar to access this feature.")
    else:
        st.subheader("Generate 3D Model from Image")
        uploaded_model_img = st.file_uploader("Upload Image for 3D Model Generation", type=["png", "jpg", "jpeg", "bmp"])

        if uploaded_model_img:
            image = Image.open(uploaded_model_img)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.form(key='model_generation_form'):
                texture_resolution = st.selectbox("Texture Resolution:", ["512", "1024", "2048"])
                foreground_ratio = st.slider("Foreground Ratio:", 0.1, 1.0, 0.85, 0.01)
                remesh = st.selectbox("Remesh:", ["none", "quad", "triangle"])
                vertex_count = st.number_input("Vertex Count (-1 for default):", min_value=-1, max_value=20000, value=-1, step=1)
                submit_model = st.form_submit_button("Generate 3D Model")

            if submit_model:
                def generate_3d_model():
                    st.info("Generating 3D model...")
                    data = {
                        "texture_resolution": texture_resolution,
                        "foreground_ratio": foreground_ratio,
                        "remesh": remesh,
                        "vertex_count": vertex_count
                    }
                    headers = {
                        "Authorization": f"Bearer {st.session_state['api_key']}",
                    }
                    url = "https://api.stability.ai/v2beta/3d/stable-fast-3d"

                    files = {
                        "image": uploaded_model_img.read(),
                    }

                    response = requests.post(url, headers=headers, files=files, data=data)
                    if response.status_code == 200:
                        glb_data = response.content
                        filename = f"generated_model_{int(time.time())}.glb"
                        save_file(glb_data, filename)
                        st.success("3D Model generated successfully!")
                        display_3d_model(glb_data, caption="Generated 3D Model")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")

                generate_3d_model()

# --- Tab 5: File Management ---
with tab5:
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

