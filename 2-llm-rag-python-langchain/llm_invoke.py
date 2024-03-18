import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI

llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

response = llm.invoke("What is Neo4j?")

print(response)