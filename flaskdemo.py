from flask import Flask

app = Flask(__name__)

import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core import load_index_from_storage

# NOTE: for local testing only, do NOT deploy with your key hardcoded

#index = None
index_name = "./saved_index"

def initialize_index():
    global index
    storage_context = StorageContext.from_defaults()
    if os.path.exists(index_name):
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        storage_context.persist(index_name)
initialize_index()
from flask import request

@app.route("/query", methods=["GET"])
def query_index():
    global index
    query_text = request.args.get("text", None)
    if query_text is None:
        return (
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return str(response), 200

@app.route("/")
def home():
    return "Hello World!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)

