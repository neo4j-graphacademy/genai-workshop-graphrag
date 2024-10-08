import asyncio
import logging.config

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
# tag::text_splitter[]
text_splitter = FixedSizeSplitter(chunk_size=150, chunk_overlap=20)
# end::text_splitter[]


# 2. Embed the chunks
# tag::embedder[]
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
# end::embedder[]


# 3. List entities and relationships to extract
# tag::schema[]
entities = ["Person", "House", "Planet", "Organization"]
relations = ["PARENT_OF", "HEIR_OF", "RULES", "MEMBER_OF"]
potential_schema = [
    ("Person", "PARENT_OF", "Person"),
    ("Person", "HEIR_OF", "House"),
    ("House", "RULES", "Planet"),
    ("Person", "RULES", "Organization"),
]
# end::schema[]


# 4. Extract nodes and relationships from the chunks
# tag::llm[]
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    },
)
# end::llm[]


# 5. Create the pipeline
# tag::create_pipeline[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
    entities=entities,
    text_splitter=text_splitter,
    relations=relations,
    on_error="IGNORE",
    from_pdf=False,
)
# end::create_pipeline[]


# 6. Run the pipeline
# tag::run_pipeline[]
asyncio.run(
    kg_builder.run_async(
        text=(
            "The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of "
            "House Atreides, an aristocratic family that rules the planet Caladan. Lady "
            "Jessica is a Bene Gesserit and an important key in the Bene Gesserit breeding "
            "program."
        )
    )
)
driver.close()
# end::run_pipeline[]
