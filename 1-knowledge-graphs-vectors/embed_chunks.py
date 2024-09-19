import os

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "sk-..." # !!!! ADD YOUR OPENAI API KEY HERE !!!
# We need an embeddings model in order to create embeddings from our chunks
# We can use the OpenAI text-embedding-3-large model
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
# Create an embedding for the first sentence of the Wikipedia page for London
chunk = "London is the capital and largest city of both England and the United Kingdom, with a population of 8,866,180 in 2022."
print(embedder.embed_query(chunk))
