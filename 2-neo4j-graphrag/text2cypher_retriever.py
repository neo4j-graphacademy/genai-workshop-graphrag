import os
import logging.config

from neo4j import GraphDatabase

from dotenv import load_dotenv
load_dotenv()

# Connect to Neo4j database
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))


# 1. Initialize the Text2CypherRetriever


# 2. Using the Retriever


driver.close()
