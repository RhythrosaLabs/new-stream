# agent.py

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from tools import get_tools
from langchain.schema import SystemMessage

def initialize_chat_agent(system_message: SystemMessage, vectorstore, openai_api_key: str):
    tools = get_tools(vectorstore, openai_api_key)

    # Initialize the agent's memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.chat_memory.add_message(system_message)

    # Agent configuration
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")]
    }
    llm = ChatOpenAI(model_name="gpt-4o-mini", streaming=True, openai_api_key=openai_api_key)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        memory=memory,
        agent_kwargs=agent_kwargs,
    )

    return agent, memory
