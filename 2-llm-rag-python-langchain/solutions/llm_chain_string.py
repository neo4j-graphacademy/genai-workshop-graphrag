import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
# tag::import_str_parser[]
from langchain.schema import StrOutputParser
# end::import_str_parser[]

llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

template = PromptTemplate.from_template("""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""")

# tag::llm_chain[]
llm_chain = template | llm | StrOutputParser()

# end::llm_chain[]

response = llm_chain.invoke({"fruit": "apple"})

print(response)