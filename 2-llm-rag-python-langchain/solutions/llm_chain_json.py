import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# tag::import_json_parser[]
from langchain.output_parsers.json import SimpleJsonOutputParser
# end::import_json_parser[]

llm = OpenAI(os.getenv('OPENAI_API_KEY'))

# tag::prompt[]
template = PromptTemplate.from_template("""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Output JSON as {{"description": "your response here"}}

Tell me about the following fruit: {fruit}
""")
# end::prompt[]

# tag::llm_chain[]
llm_chain = LLMChain(
    llm=llm,
    prompt=template,
    output_parser=SimpleJsonOutputParser()
)
# end::llm_chain[]

response = llm_chain.invoke({"fruit": "apple"})

print(response)