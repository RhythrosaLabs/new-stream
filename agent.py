# agent.py

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.callbacks import StreamlitCallbackHandler

def initialize_langchain_agent(tools, model_name="gpt-4o-mini"):
    """
    Initialize the LangChain agent with the provided tools and memory.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")]
    }
    llm = ChatOpenAI(model_name=model_name, streaming=True)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        memory=memory,
        agent_kwargs=agent_kwargs,
    )
    return agent, memory