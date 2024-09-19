import asyncio
import logging.config
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.llm import OpenAILLM

load_dotenv()

# Set log level to DEBUG for all neo4j_graphrag.* loggers
logging.config.dictConfig(
    {
        "version": 1,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
            }
        },
        "loggers": {
            "root": {
                "handlers": ["console"],
            },
            "neo4j_graphrag": {
                "level": "DEBUG",
            },
        },
    }
)

# ================================
# === 1. Creating the Pipeline ===
# ================================

# First we create the pipeline.
# The pipeline allows us to chain various components, such as text splitters, embedders,
# and LLM graph extractors together in order to build a knowledge graph
# tag::create_pipeline[]
pipe = Pipeline()
# end::create_pipeline[]


# ============================
# === 2. Chunking the Text ===
# ============================

# The first step in the pipeline is to split the text into chunks.
# We create a text splitter component then add it to the pipeline.
# tag::split_text[]
text_splitter = FixedSizeSplitter(chunk_size=200, chunk_overlap=20)
pipe.add_component(text_splitter, "text_splitter")
# end::split_text[]


# ===============================
# === 3. Embedding the Chunks ===
# ===============================

# The next step is to create an embedding for each chunk.
# We create a chunk embedder component then add it to the pipeline.
# We then connect the text splitter to the embedder to let the pipeline know we want to
# feed the chunks created by the text splitter into the embedder.
# tag::embed_chunks[]
embedder = TextChunkEmbedder(embedder=OpenAIEmbeddings(model="text-embedding-3-large"))
pipe.add_component(embedder, "embedder")
pipe.connect("text_splitter", "embedder", input_config={"text_chunks": "text_splitter"})
# end::embed_chunks[]

# =============================================================
# === 4. Extracting Nodes and Relationships from the Chunks ===
# =============================================================

# Now we use an LLM to extract nodes and relationships from each chunk.
# We create a LLM extractor component then add it to the pipeline.
# We then connect the embedder to the LLM extractor component to let the pipeline know
# we want to feed the chunks and their embeddings created by the embedder into the LLM.
# We need an embeddings model in order to create embeddings from our chunks.
# We can use the OpenAI text-embedding-3-large model for this.

# tag::use_llm_extractor[]
llm = LLMEntityRelationExtractor(
    llm=OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "response_format": {"type": "json_object"},
        },
    ),
    on_error=OnError.RAISE,
    max_concurrency=1,
)
pipe.add_component(llm, "llm")
pipe.connect("embedder", "llm", input_config={"chunks": "embedder"})
# end::use_llm_extractor[]


# =======================================
# === 5. Creating the Knowledge Graph ===
# =======================================

# Finally we upload everything to Neo4j in order to build our knowledge graph.
# We first Neo4j driver instance to connect to our database.
# We create a Neo4j write component then add it to the pipeline.
# We then connect to LLM extractor to the Neo4j writer component to let the pipeline
# know we want to write the nodes, relationships, chunks, and embeddings we extracted in
# previous steps to the Neo4j database.
# tag::write_graph[]
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
driver = GraphDatabase.driver(uri=URI, auth=AUTH)
writer = Neo4jWriter(driver, max_concurrency=1)
pipe.add_component(writer, "writer")
pipe.connect(
    "llm",
    "writer",
    input_config={"graph": "llm"},
)
# end::write_graph[]


# ===============================
# === 6. Running the Pipeline ===
# ===============================

# Finally we feed our input to the pipeline then run it to create our knowledge graph
# tag::run_pipeline[]
text = """
London is the capital and largest city of both England and the United Kingdom, with a
population of 8,866,180 in 2022. The wider metropolitan area is the largest in Western
Europe, with a population of 14.9 million. London stands on the River Thames in
southeast England, at the head of a 50-mile (80 km) estuary down to the North Sea, and
has been a major settlement for nearly 2,000 years. Its ancient core and financial
centre, the City of London, was founded by the Romans as Londinium and has retained its
medieval boundaries. The City of Westminster, to the west of the City of London, has
been the centuries-long host of the national government and parliament. London grew
rapidly in the 19th century, becoming the world's largest city at the time. Since the
19th century, the name "London" has referred to the metropolis around the City of
London, historically split between the counties of Middlesex, Essex, Surrey, Kent, and
Hertfordshire, which since 1965 has largely comprised the administrative area of Greater
London, governed by 33 local authorities and the Greater London Authority.
"""
asyncio.run(pipe.run({"text_splitter": {"text": text}}))
driver.close()
# end::run_pipeline[]

