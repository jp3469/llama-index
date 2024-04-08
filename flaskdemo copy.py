from flask import Flask

app = Flask(__name__)

import os
import openai
import nest_asyncio

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core import load_index_from_storage
from llama_index.readers.file import UnstructuredReader
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.core import load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.agent.openai import OpenAIAgent

from dotenv import load_dotenv
load_dotenv()
# NOTE: for local testing only, do NOT deploy with your key hardcoded

#index = None
index_name = "./saved_index"

def initialize_index():
    global agent
    openai.api_key = os.environ.get("OPENAI_API_KEY")


    nest_asyncio.apply()

    years = [2020]
    import nltk
    nltk.download('averaged_perceptron_tagger')
    loader = UnstructuredReader()
    doc_set = {}
    all_docs = []
    for year in years:
        year_docs = loader.load_data(
            file=Path(f"./data/UBER/UBER_{year}.html"), split_documents=False
        )
        # insert year metadata into each year
        for d in year_docs:
            d.metadata = {"year": year}
        doc_set[year] = year_docs
        all_docs.extend(year_docs)

    # initialize simple vector indices

    Settings.chunk_size = 512
    index_set = {}
    for year in years:
        storage_context = StorageContext.from_defaults()
        cur_index = VectorStoreIndex.from_documents(
            doc_set[year],
            storage_context=storage_context,
        )
        index_set[year] = cur_index
        storage_context.persist(persist_dir=f"./storage/{year}")

    # Load indices from disk

    index_set = {}
    for year in years:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./storage/{year}"
        )
        cur_index = load_index_from_storage(
            storage_context,
        )
        index_set[year] = cur_index


    individual_query_engine_tools = [
        QueryEngineTool(
            query_engine=index_set[year].as_query_engine(),
            metadata=ToolMetadata(
                name=f"vector_index_{year}",
                description=f"useful for when you want to answer queries about the {year} SEC 10-K for Uber",
            ),
        )
        for year in years
    ]

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=individual_query_engine_tools,
        llm=OpenAI(model="gpt-3.5-turbo"),
    )

    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="sub_question_query_engine",
            description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber",
        ),
    )

    tools = individual_query_engine_tools + [query_engine_tool]


    agent = OpenAIAgent.from_tools(tools)
initialize_index()
from flask import request

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

