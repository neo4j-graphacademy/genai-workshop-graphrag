from neo4j import GraphDatabase

# Demo database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")

# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)

import os
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "sk-…"
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM

# Create LLM object
t2c_llm = OpenAILLM(model_name="gpt-3.5-turbo")

# (Optional) Specify your own Neo4j schema
neo4j_schema = """
Node properties:
Person {name: STRING, born: INTEGER}
Movie {tagline: STRING, title: STRING, released: INTEGER}
Relationship properties:
ACTED_IN {roles: LIST}
REVIEWED {summary: STRING, rating: INTEGER}
The relationships:
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)
(:Person)-[:PRODUCED]->(:Movie)
(:Person)-[:WROTE]->(:Movie)
(:Person)-[:FOLLOWS]->(:Person)
(:Person)-[:REVIEWED]->(:Movie)
"""

# Provide user input/query pairs for the LLM to use as examples
examples = [
    "USER INPUT: 'Which actors starred in the Matrix?' QUERY: MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE m.title = 'The Matrix' RETURN p.name"
]

# Build the retriever
# retriever = Text2CypherRetriever()

from neo4j_graphrag.generation import GraphRAG

llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
rag = GraphRAG(retriever=retriever, llm=llm)

query_text = "Which movies did Hugo Weaving star in?"
response = rag.search(query_text=query_text)
print(response.answer)
