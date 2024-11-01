# tools.py

import openai
import requests
import base64
from langchain.tools import Tool, DuckDuckGoSearchRun
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def create_web_search_tool():
    """
    Create the web search tool using DuckDuckGo.
    """
    return Tool(
        name="web_search",
        func=DuckDuckGoSearchRun().run,
        description="Useful for answering questions about current events or the internet."
    )

def create_openai_image_generation_tool(openai_api_key):
    """
    Create the image generation tool using OpenAI's DALL·E 3.
    """

    def generate_image(prompt):
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024",
                response_format="url",
                model="dall-e-3"
            )
            image_url = response['data'][0]['url']
            return image_url
        except Exception as e:
            return f"Error generating image with OpenAI: {e}"

    return Tool(
        name="openai_image_generation",
        func=generate_image,
        description="Generates an image based on the prompt using OpenAI's DALL·E 3."
    )

def create_stability_image_generation_tool(stability_api_key):
    """
    Create the image generation tool using Stability AI's Stable Diffusion.
    """

    def generate_image(prompt):
        try:
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/ultra",
                headers={
                    "Authorization": f"Bearer {stability_api_key}",
                    "Accept": "application/json"
                },
                data={
                    "prompt": prompt,
                    "output_format": "png"
                },
                files={"none": ''}
            )
            if response.status_code == 200:
                data = response.json()
                # Assuming the response contains 'artifacts' with 'base64'
                image_base64 = data['artifacts'][0]['base64']
                return f"data:image/png;base64,{image_base64}"
            else:
                return f"Error generating image with Stability AI: {response.text}"
        except Exception as e:
            return f"Error generating image with Stability AI: {e}"

    return Tool(
        name="stability_image_generation",
        func=generate_image,
        description="Generates an image based on the prompt using Stability AI's Stable Diffusion Ultra."
    )

def create_stability_3d_generation_tool(stability_api_key):
    """
    Create the 3D generation tool using Stability AI's Stable Fast 3D.
    """

    def generate_3d_model(prompt):
        try:
            # Generate an image from the prompt using Stability AI's image generation
            image_response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/ultra",
                headers={
                    "Authorization": f"Bearer {stability_api_key}",
                    "Accept": "application/json"
                },
                data={
                    "prompt": prompt,
                    "output_format": "png"
                },
                files={"none": ''}
            )
            if image_response.status_code != 200:
                return f"Error generating image for 3D model: {image_response.text}"
            image_data = image_response.json()['artifacts'][0]['base64']
            # Decode the base64 image
            image_bytes = base64.b64decode(image_data)
            
            # Send the image to Stability AI's 3D generation API
            response = requests.post(
                "https://api.stability.ai/v2beta/3d/stable-fast-3d",
                headers={
                    "Authorization": f"Bearer {stability_api_key}"
                },
                files={
                    "image": image_bytes
                },
                data={}
            )
            if response.status_code == 200:
                glb_data = response.content
                glb_base64 = base64.b64encode(glb_data).decode('utf-8')
                return f"data:model/gltf-binary;base64,{glb_base64}"
            else:
                return f"Error generating 3D model: {response.text}"
        except Exception as e:
            return f"Error generating 3D model: {e}"

    return Tool(
        name="stability_3d_generation",
        func=generate_3d_model,
        description="Generates a 3D model based on the prompt using Stability AI's Stable Fast 3D."
    )

def create_stability_video_generation_tool(stability_api_key):
    """
    Create the video generation tool using Stability AI's Image-to-Video.
    """

    def generate_video(prompt):
        try:
            # Generate an image from the prompt using Stability AI's image generation
            image_response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/ultra",
                headers={
                    "Authorization": f"Bearer {stability_api_key}",
                    "Accept": "application/json"
                },
                data={
                    "prompt": prompt,
                    "output_format": "png"
                },
                files={"none": ''}
            )
            if image_response.status_code != 200:
                return f"Error generating image for video: {image_response.text}"
            image_data = image_response.json()['artifacts'][0]['base64']
            # Decode the base64 image
            image_bytes = base64.b64decode(image_data)
            
            # Start the video generation process
            start_response = requests.post(
                "https://api.stability.ai/v2beta/image-to-video",
                headers={
                    "Authorization": f"Bearer {stability_api_key}"
                },
                files={
                    "image": image_bytes
                },
                data={
                    "cfg_scale": 1.8,
                    "motion_bucket_id": 127
                }
            )
            if start_response.status_code != 200:
                return f"Error starting video generation: {start_response.text}"
            generation_id = start_response.json().get('id')
            if not generation_id:
                return "Error: No generation ID returned."
            
            # Poll for the result
            import time
            max_polls = 10
            poll_interval = 10  # seconds
            for _ in range(max_polls):
                time.sleep(poll_interval)
                result_response = requests.get(
                    f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}",
                    headers={
                        "Authorization": f"Bearer {stability_api_key}",
                        "Accept": "video/mp4"  # Request video directly
                    }
                )
                if result_response.status_code == 200:
                    video_data = result_response.content
                    video_base64 = base64.b64encode(video_data).decode('utf-8')
                    return f"data:video/mp4;base64,{video_base64}"
                elif result_response.status_code == 202:
                    st.write("Video generation in progress, waiting...")
                else:
                    return f"Error fetching video generation result: {result_response.text}"
            return "Video generation timed out."

    return Tool(
        name="stability_video_generation",
        func=generate_video,
        description="Generates a video based on the prompt using Stability AI's Image-to-Video."
    )

def create_document_qa_tool(vectorstore, openai_api_key):
    """
    Create the document Q&A tool using LangChain's RetrievalQA.
    """

    def answer_question_about_document(question):
        if vectorstore is None:
            return "No document has been uploaded. Please upload a document to use this feature."
        try:
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key),
                chain_type="stuff",
                retriever=retriever
            )
            answer = qa_chain.run(question)
            return answer
        except Exception as e:
            return f"Error answering question about the document: {e}"

    return Tool(
        name="document_qa",
        func=answer_question_about_document,
        description=(
            "Use this tool to answer any questions related to the content of the uploaded document. "
            "If the user asks about the document, its content, or any topics covered in the document, "
            "you should use this tool to provide an accurate answer."
        )
    )
