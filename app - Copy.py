
import os   
PATH = 'D:\ChatBot\cache'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
from flask import Flask, request
import json
from main import find_most_similar_batch



app = Flask(__name__)

@app.route("/")
def home_route():
    return("hello from Home Page")

@app.route("/get")
def get_bot_response():
    print('hit url')
    userText = request.args.get('msg')
    response = find_most_similar_batch(userText)
    print("give res")
    return json.dumps(response, indent=2)

if __name__ == "__main__":

    app.run(debug=True, use_reloader = False)