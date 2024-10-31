# tools.py

import openai
from langchain.tools import Tool
from langchain.utilities import DuckDuckGoSearchRun
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

def create_image_generation_tool(openai_api_key):
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
            return f"Error generating image: {e}"

    return Tool(
        name="image_generation",
        func=generate_image,
        description="Generates an image based on the prompt."
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
