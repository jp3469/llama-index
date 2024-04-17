"""
YELP API KEY
Client ID

# ADD YOUR OWN PLEASE
I0X5aql7nxiWsM2gwQKOQg 


API Key
"""


import os
from dotenv import load_dotenv

load_dotenv() 
# ADD YOUR OWN PLEASE
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

import json
from typing import Sequence, List

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool

import nest_asyncio
import requests
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from yelpapi import YelpAPI

import googlemaps

yelp_api_key = os.environ.get("YELP_API_KEY")

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")


# Define our tools
# #1 tool, get top 10 restaurants in a location based on a query
def restaurant_search(
    query: str,
    location: str,
    api_key: str,
    limit: int = 5,
    sort_by: str = "rating",
) -> List[dict]:
    yelp_api = YelpAPI(yelp_api_key)
    response = yelp_api.search_query(
        term=query, location=location, sort_by=sort_by, limit=limit
    )
    return response["businesses"]

# #2 tool, given a business id or alias, get the details of a restaurant
def restaurant_details_search(
    business_id_or_alias: str,
) -> List[dict]:
    yelp_api = YelpAPI(yelp_api_key)
    response = yelp_api.business_query(id=business_id_or_alias)
    return response

# #3 tool, given the location of a restaurant, get the distance and travel time to and from your current location to the restaurant
def distance_to_restaurant(
        curr_location: str,
        restaurant_location: str
    ):
        map_client = googlemaps.Client(GOOGLE_MAPS_API_KEY)
        response = map_client.distance_matrix(destinations = restaurant_location, origins = curr_location)
        return response

restaurant_search_tool = FunctionTool.from_defaults(fn=restaurant_search)
restaurant_details_tool = FunctionTool.from_defaults(fn=restaurant_details_search)
distance_to_restaurant_tool = FunctionTool.from_defaults(fn=distance_to_restaurant)

# Define our chat store
# chat_store = SimpleChatStore.from_persist_path(
#     persist_path="chat_store.json"
# )
chat_store = SimpleChatStore()

chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=10000,
    chat_store=chat_store,
    chat_store_key="user1",
)


llm = OpenAI(model="gpt-3.5-turbo-1106")
agent = OpenAIAgent.from_tools(
    [restaurant_search_tool, restaurant_details_tool, distance_to_restaurant_tool], llm=llm, memory=chat_memory
)


# while True:
#     user_input = input("You: ")
#     if user_input == "exit":
#         # import json
#         # with open('chat_store.json', 'w') as f:
#         #     json.dump(chat_store.json(), f)
#         break
#     response = agent.chat(user_input)
#     print(f"Bot: {response}")


from flask import request
from flask import Flask
app = Flask(__name__)

@app.route("/query", methods=["GET"])
def query_index():
    # global index
    query_text = request.args.get("text", None)
    if query_text is None:
        return (
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    # query_engine = index.as_query_engine()
    # response = query_engine.query(query_text)
    response = agent.chat(query_text)
    return str(response), 200

@app.route("/")
def home():
    return "Hello World!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)