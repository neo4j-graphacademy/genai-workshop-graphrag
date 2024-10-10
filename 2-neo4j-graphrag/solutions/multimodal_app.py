import os

import neo4j
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem
from dotenv import load_dotenv
load_dotenv()


uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))


# tag::embedder[]
IMAGE_EMBEDDING_MODEL = "clip-ViT-B-32"
embedder = SentenceTransformerEmbeddings(IMAGE_EMBEDDING_MODEL)
# end::embedder[]

# tag::retriever[]
POSTER_INDEX_NAME = "moviePosters"
retrieval_query = "RETURN node.title as title, node.plot as plot, node.poster as posterUrl, score"

def format_record_function(record: neo4j.Record) -> RetrieverResultItem:
    return RetrieverResultItem(
        content=f"Movie title: {record.get('title')}, movie plot: {record.get('plot')}",
        metadata={
            "title": record.get("title"),
            "plot": record.get("plot"),
            "poster": record.get("posterUrl"),
            "score": record.get("score"),
        },
    )

retriever = VectorCypherRetriever(
    driver,
    index_name=POSTER_INDEX_NAME,
    retrieval_query=retrieval_query,
    result_formatter=format_record_function,
    embedder=embedder,
)

query_text = ("Find a movie where in the poster there are only animals without people")
top_k = 3

result = retriever.search(query_text=query_text, top_k=top_k)

for r in result.items:
    print(r.content, r.metadata.get("score"))
    print(r.metadata["poster"])
# end::retriever[]
driver.close()
