import os
from dotenv import load_dotenv
load_dotenv()

# tag::import_openaiembeddings[]
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# end::import_openaiembeddings[]
from langchain.prompts import PromptTemplate
# tag::import_retrievalqa[]
from langchain.chains import LLMChain, RetrievalQA
# end::import_retrievalqa[]
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_community.tools import YouTubeSearchTool
# tag::import_vector[]
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
# end::import_vector[]

llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

# tag::embedding_provider[]
embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY')
    )
# end::embedding_provider[]

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

youtube = YouTubeSearchTool()

# tag::movie_plot_vector[]
movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    index_name="moviePlots",
    embedding_node_property="embedding",
    text_node_property="plot",
)
# end::movie_plot_vector[]

# tag::retriever[]
plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever()
)
# end::retriever[]

# tag::run_retriever[]
def run_retriever(query):
    results = plot_retriever.invoke({"query":query})
    return str(results)
# end::run_retriever[]

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
    Tool.from_function(
        name="Movie Plot Search",
        description="For when you need to compare a plot to a movie. The question will be a string. Return a string.",
        func=run_retriever,
        return_direct=True  
    )
]
# end::tools[]

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

while True:
    q = input("> ")
    response = agent_executor.invoke({"input": q})
    print(response["output"])
