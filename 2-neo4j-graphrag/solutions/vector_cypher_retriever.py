# tag::setup[]
from neo4j import GraphDatabase
uri = "neo4j+s://demo.neo4jlabs.com"
username = "recommendations"
password = "recommendations"
driver = GraphDatabase.driver(uri, auth=(username, password))
# end::setup[]

# tag::embedder[]
import os
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "sk-…"
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
# end::embedder[]

# tag::retriever[]
from neo4j_graphrag.retrievers import VectorCypherRetriever

retrieval_query = """
MATCH
(actor:Actor)-[:ACTED_IN]->(node)
RETURN
node.title AS movie_title,
node.plot AS movie_plot,
collect(actor.name) AS actors;
"""

retriever = VectorCypherRetriever(
    driver,
    index_name="moviePlotsEmbedding",
    embedder=embedder,
    retrieval_query=retrieval_query,
)
# end::retriever[]

# tag::graphrag[]
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM

llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
rag = GraphRAG(retriever=retriever, llm=llm)
query_text = "Who were the actors in the movie about the magic jungle board game?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
print(response.answer)
# end::graphrag[]
