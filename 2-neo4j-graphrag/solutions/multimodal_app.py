import os

import neo4j
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

uri = "neo4j+s://demo.neo4jlabs.com"
username = "recommendations"
password = "recommendations"
driver = GraphDatabase.driver(uri, auth=(username, password))


os.environ["OPENAI_API_KEY"] = "sk-…"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

POSTER_INDEX_NAME = "moviePostersEmbedding"
IMAGE_EMBEDDING_MODEL = "clip-ViT-B-32"


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


embedder = SentenceTransformerEmbeddings(IMAGE_EMBEDDING_MODEL)
retrieval_query = "RETURN node.title as title, node.plot as plot, node.poster as posterUrl, score"
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

driver.close()
