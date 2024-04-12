
import os   
PATH = 'C:\\WebApp\\ChatBot_New\\ChatBot\\Transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
from flask import Flask, request,jsonify
import json
from main import find_most_similar_batch,upload_file



app = Flask(__name__)

@app.route("/")
def home_route():
    return("hello from Home Page")

@app.route("/upload",methods=["POST"])
def file_uploaded():
    request_data = request.json
    new_data = {
        "anchor_text": request_data.get("anchor_text"),
        "anchor_link": request_data.get("anchor_link"),
        "parent_url": request_data.get("parent_url"),
        "scripted_value": request_data.get("scripted_value", "")
    }
    temp = upload_file(new_data)

    if(temp):
     return jsonify({"message": "Data uploaded successfully."}), 200

  

@app.route("/get")
def get_bot_response():
    print('hit url')
    userText = request.args.get('msg')
    response = find_most_similar_batch(userText)
    print("give res")
    return json.dumps(response, indent=2)

if __name__ == "__main__":

    app.run(debug=True, use_reloader = False)