import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

chat_llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

# tag::prompt[]
prompt = PromptTemplate(
    template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Context: {context}
Question: {question}
""",
    input_variables=["context", "question"],
)
# end::prompt[]

# tag::context[]
current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""
# end::context[]

chat_chain = LLMChain(
    llm=chat_llm, 
    prompt=prompt
    )

# tag::invoke[]
response = chat_chain.invoke(
    {
        "context": current_weather,
        "question": "What is the weather like on Watergate Bay?"
    }
)
# end::invoke[]

print(response)
