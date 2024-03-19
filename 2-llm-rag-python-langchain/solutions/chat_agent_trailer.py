import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
# tag::import_youtube[]
from langchain_community.tools import YouTubeSearchTool
# end::import_youtube[]

llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

prompt = PromptTemplate(
    template="""
    You are a movie expert. You find movies from a genre or plot.

    Chat History:{chat_history}
    Question:{input}
    """,
    input_variables=["chat_history", "input"],
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# tag::youtube[]
youtube = YouTubeSearchTool()
# end::youtube[]

# tag::tools[]
tools = [
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=chat_chain.run,
        return_direct=True,
    ),
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word 'trailer'. Return a link to a YouTube video.",
        func=youtube.run,
        return_direct=True,
    ),
]
# end::tools[]

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

while True:
    q = input("> ")
    response = agent_executor.invoke({"input": q})
    print(response["output"])
