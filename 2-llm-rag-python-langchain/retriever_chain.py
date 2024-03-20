import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    index_name="moviePlots",
    embedding_node_property="embedding",
    text_node_property="plot",
)

plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever()
)

result = plot_retriever.invoke(
    {"query": "A movie where a mission to the moon goes wrong"}
)

print(result)