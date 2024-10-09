import asyncio
import logging.config
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.openai_llm import OpenAILLM

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

# Connect to the Neo4j database
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
driver = GraphDatabase.driver(URI, auth=AUTH)


# 1. Chunk the text


# 2. Embed the chunks


# 3. List entities and relationships to extract


# 4. Extract nodes and relationships from the chunks


# 5. Create the pipeline


# 6. Run the pipeline


driver.close()
