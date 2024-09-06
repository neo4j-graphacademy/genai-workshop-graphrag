import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
# tag::imports[]
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from uuid import uuid4
# end::imports[]
# tag::session-id[]
SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")
# end::session-id[]

chat_llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

# tag::neo4j-graph[]
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
)
# end::neo4j-graph[]

# tag::get-memory[]
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)
# end::get-memory[]

# tag::prompt[]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
# end::prompt[]


# tag::chat-history[]
chat_chain = prompt | chat_llm | StrOutputParser()

chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)
# end::chat-history[]

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Bells", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

# tag::loop[]
while True:
    question = input("> ")

    response = chat_with_message_history.invoke(
        {
            "context": current_weather,
            "question": question,
            
        }, 
        config={
            "configurable": {"session_id": SESSION_ID}
        }
    )
    
    print(response)
# end::loop[]