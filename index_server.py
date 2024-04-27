import os
import pickle

from multiprocessing import Lock
from multiprocessing.managers import BaseManager

from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.agent.openai import OpenAIAgent
from typing import Sequence, List
from dotenv import load_dotenv
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

load_dotenv()
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
yelp_api_key = os.environ.get("YELP_API_KEY")

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

def initialize_agent():
    """Create a new global index, or load one from the pre-set path."""
    global agent
    # NOTE: for local testing only, do NOT deploy with your key hardcoded
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
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
    prompt = """You are a restaurant recommending chat bot. I am a user that needs help picking out a good restaurant to eat. First, if the user asks where they should eat, you should collect the following information with as many questions and answers as possible, be thorough and ask the user carefully:
                Cuisine -- what the user likes and dislikes
                price range -- in $ value, one to three dollar signs
                location -- current location and how far they are willing to travel
                allergies

                After collecting that information, if the user is unsure of what cuisine they want to eat, ask a few more questions to narrow down the options such as:
                Are you in the mood for a healthier option or are you in the mood for a more indulgent treat?
                Are you in the mood for familiar comfort food or are you open to trying something new?
                Feel free to ask more questions to find the best options for the user.

                Using the data you have collected, provide a list of 5 restaurants within the location range so that: it follows the cuisine they like and not a cuisine they dislike, follows specified dietary restrictions if any an falls within the similar price range, and should not be a restaurant they have eaten recently.
                Your output should be a list of json with following format
                {
                restaurantName: string,
                price range: string,
                location: string,
                recommendationReason: string
                }

                If the user does not like any of the restaurant options provided above, provide a new list following the same guidelines as before in the same format.
                If the user is unable to decide which one of the listed restaurants to choose, offer to play a game with the user to help decide. Ask if the user wants to play a game that will narrow down the options based on preferences or a pure luck based game.
                If the user asks to play a game based on preferences, play a "this or that" game where the user must select from two options and the options are characteristics of different restaurants on the list and narrow down the list.
                IF the user asks to play a game based on chance, play any randomizing game to help the user decide.

                If the user answer with unnecessary information, stick to the script and follow the flow"""
    agent = OpenAIAgent.from_tools(
        [restaurant_search_tool, restaurant_details_tool, distance_to_restaurant_tool], llm=llm, memory=chat_memory, system_prompt=prompt
    )


def query_agent(query_text):
    """Ask the agent a question."""
    global agent
    response = agent.chat(query_text)
    return response


if __name__ == "__main__":
    # init the global index
    print("initializing index...")
    initialize_agent()

    # setup server
    # NOTE: you might want to handle the password in a less hardcoded way
    manager = BaseManager(('', 5602), b'password')
    manager.register('query_agent', query_agent)
    server = manager.get_server()

    print("server started...")
    server.serve_forever()