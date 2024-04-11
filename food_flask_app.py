import os
from multiprocessing.managers import BaseManager
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# initialize manager connection
# NOTE: you might want to handle the password in a less hardcoded way
manager = BaseManager(('', 5602), b'password')
manager.register('query_agent')
manager.connect()


@app.route("/query", methods=["GET"])
def query_agent():
    global manager
    query_text = request.args.get("text", None)
    if query_text is None:
        return "No text found, please include a ?text=blah parameter in the URL", 400
    
    response = manager.query_agent(query_text)
    return make_response(jsonify(response)), 200

@app.route("/")
def home():
    return "Hello, World! Welcome to the food chatbot!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)