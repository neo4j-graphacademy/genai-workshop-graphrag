import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

# Demo database credentials
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))

# tag::embedder[]
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
# end::embedder[]

# tag::retriever[]
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.llm import OpenAILLM

retrieval_query = """
MATCH (actor:Actor)-[:ACTED_IN]->(node:Movie)
RETURN node.title AS movie_title,
       node.plot AS movie_plot,
       collect(actor.name) AS actors;
"""

retriever = HybridCypherRetriever(
    driver=driver,
    vector_index_name="moviePlots",
    fulltext_index_name="plotFulltext",
    retrieval_query=retrieval_query,
    embedder=embedder,
)
# end::retriever[]

# tag::graphrag[]
from neo4j_graphrag.generation import GraphRAG

llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
rag = GraphRAG(retriever=retriever, llm=llm)
query_text = "What are the names of the actors in the movie set in 1375 in Imperial China?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
print(response.answer)
# end::graphrag[]
driver.close()
