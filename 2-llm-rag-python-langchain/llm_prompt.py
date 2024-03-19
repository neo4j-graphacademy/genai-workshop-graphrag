import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

# Create prompt template
# template =

# Invoke the llm using the prompt template
# response = 
