import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser
from langchain_community.tools import YouTubeSearchTool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from uuid import uuid4

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a movie expert. You find movies from a genre or plot.",
        ),
        ("human", "{input}"),
    ]
)

movie_chat = prompt | llm | StrOutputParser()

youtube = YouTubeSearchTool()

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever()
)

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

def call_trailer_search(input):
    input = input.replace(",", " ")
    return youtube.run(input)

tools = [
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=movie_chat.invoke,
    ),
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word trailer. Return a link to a YouTube video.",
        func=call_trailer_search,
    ),
    Tool.from_function(
        name="Movie Plot Search",
        description="For when you need to compare a plot to a movie. The question will be a string. Return a string.",
        func=plot_retriever.invoke,
    ),
]

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while True:
    q = input("> ")

    response = chat_agent.invoke(
        {
            "input": q
        },
        {"configurable": {"session_id": SESSION_ID}},
    )
    
    print(response["output"])
