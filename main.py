import streamlit as st
import openai
import anthropic
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="Combined AI App", page_icon="🤖")

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
    anthropic_api_key = st.text_input("Anthropic API Key", key="anthropic_api_key", type="password")
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
    st.markdown("[View the source code](https://github.com/streamlit/llm-examples/blob/main)")
    st.markdown("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)")

st.title("🤖 Combined AI App")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📝 File Q&A", "🔎 Chat with Search"])

# Tab 1: Chatbot
with tab1:
    st.header("💬 Chatbot")
    if "messages_chatbot" not in st.session_state:
        st.session_state["messages_chatbot"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state["messages_chatbot"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        openai.api_key = openai_api_key
        st.session_state["messages_chatbot"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state["messages_chatbot"]
        )
        msg = response.choices[0].message.content
        st.session_state["messages_chatbot"].append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

# Tab 2: File Q&A with Anthropic
with tab2:
    st.header("📝 File Q&A with Anthropic")
    uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))

    question = st.text_input(
        "Ask something about the article",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question and not anthropic_api_key:
        st.info("Please add your Anthropic API key to continue.")

    if uploaded_file and question and anthropic_api_key:
        article = uploaded_file.read().decode()
        prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n
        {article}\n\n\n\n{question}{anthropic.AI_PROMPT}"""

        client = anthropic.Client(api_key=anthropic_api_key)
        response = client.completions.create(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-v1",  # Use "claude-2" for Claude 2 model
            max_tokens_to_sample=100,
        )
        st.write("### Answer")
        st.write(response['completion'])

# Tab 3: Chat with Search
with tab3:
    st.header("🔎 LangChain - Chat with Search")

    """
    In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
    Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
    """

    if "messages_search" not in st.session_state:
        st.session_state["messages_search"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
        ]

    for msg in st.session_state["messages_search"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
        st.session_state["messages_search"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
        search = DuckDuckGoSearchRun(name="Search")
        search_agent = initialize_agent(
            tools=[search],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True,
        )
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = search_agent.run(prompt, callbacks=[st_cb])
            except Exception as e:
                response = f"An error occurred: {e}"
            st.session_state["messages_search"].append({"role": "assistant", "content": response})
            st.write(response)
