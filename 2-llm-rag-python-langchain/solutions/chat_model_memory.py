import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
# tag::import_memory[]
from langchain.chains.conversation.memory import ConversationBufferMemory
# end::import_memory[]

chat_llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

# tag::prompt[]
prompt = PromptTemplate(
    template="""
You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Chat History: {chat_history}
Context: {context}
Question: {question}
""",
    input_variables=["chat_history", "context", "question"],
)
# end::prompt[]

# tag::memory[]
memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="question", return_messages=True
)
# end::memory[]

# tag::chat_chain[]
chat_chain = LLMChain(
    llm=chat_llm, 
    prompt=prompt, 
    memory=memory
    )
# end::chat_chain[]

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

# tag::response[]
while True:
    question = input("> ")
    response = chat_chain.invoke(
        {
            "context": current_weather, 
            "question": question
            }
    )

    print(response["text"])
# end::response[]