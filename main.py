from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker
import json
import torch
from better_profanity import profanity

# Load the SentenceTransformer model once
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Create a SpellChecker instance once
spell = SpellChecker()

json_file_path = "alldata.json"
learn_json_file = "learn.json"
offensive_words = "offensive_words.txt"

# Load existing data from JSON once
try:
    with open(json_file_path, "r") as file:
        existing_data = json.load(file)
except:
    existing_data = []

try:
    with open(offensive_words, "r") as file:
      offensive_words = file.read().split(",")
except FileNotFoundError:
      offensive_words = [] 

try:
    with open(learn_json_file, "r") as file:
        additional_data = json.load(file)
        existing_data.extend(additional_data)
except FileNotFoundError:
    pass        



profanity.add_censor_words(offensive_words)    

# Preprocess existing data
existing_data = [
     {"anchor_text": data["anchor_text"], "anchor_link": data["anchor_link"], "scripted_value": data["scripted_value"]}
    for data in existing_data
    if data["anchor_text"].strip()
]

existing_anchor_texts = {
    ' '.join(data["anchor_text"].split("-")): data for data in existing_data
}
existing_anchor_texts_set = set(existing_anchor_texts.keys())
existing_embeddings = {
    anchor_text: model.encode([anchor_text], convert_to_tensor=True)[0]
    for anchor_text in existing_anchor_texts
}

def upload_file(uploaded_data):
    try:
        with open(learn_json_file, "r") as file:
            additional_data_upload = json.load(file)
    except FileNotFoundError:
            additional_data_upload = []

    additional_data_upload.append(uploaded_data)

    with open(learn_json_file, "w") as file:
        json.dump(additional_data_upload, file,indent=4)

    file_reader()  

    return True 



def file_reader():
    global existing_data, existing_anchor_texts, existing_anchor_texts_set, existing_embeddings
    print('again')
    try:
        with open(json_file_path, "r") as file:
            existing_data_new = json.load(file)
    except FileNotFoundError:
        existing_data_new = []

    try:
        with open(learn_json_file, "r") as file:
             additional_data = json.load(file)
             existing_data_new.extend(additional_data)
    except FileNotFoundError:
        pass      


    if len(existing_data) != len(existing_data_new):
        existing_data = existing_data_new

        existing_data = [
            {"anchor_text": data["anchor_text"],
             "anchor_link": data["anchor_link"],
             "scripted_value": data["scripted_value"]
             }
            for data in existing_data
        ]
        existing_data = [data for data in existing_data if data["anchor_text"].strip()]

        existing_anchor_texts = {
            ' '.join(data["anchor_text"].split("-")): data for data in existing_data
        }

        existing_anchor_texts_set = set(existing_anchor_texts.keys())

        existing_embeddings = {
            anchor_text: model.encode([anchor_text], convert_to_tensor=True)[0]
            for anchor_text in existing_anchor_texts
        }


def load_json(input_text, top_n, threshold):

    if len(input_text) == 1 or profanity.contains_profanity(input_text):
        return returnData([],0)
    
    corrected_tokens = [spell.correction(token) if spell.correction(token) else token for token in input_text.split()]
    corrected_input_text = ' '.join(corrected_tokens)

    if profanity.contains_profanity(input_text):
        return returnData([],0)
    
    input_embedding = model.encode([corrected_input_text], convert_to_tensor=True)[0]

    similarities = util.pytorch_cos_sim(input_embedding.unsqueeze(0), torch.stack(list(existing_embeddings.values())))
    similarities = similarities.squeeze().tolist()
    
    similarities_filtered = {
        anchor_text: score for anchor_text, score in zip(existing_embeddings.keys(), similarities) if score > threshold
    }

    top_n_anchor_texts = sorted(existing_anchor_texts_set.intersection(similarities_filtered.keys()), key=similarities_filtered.get, reverse=True)[:top_n]
    top_n_data = [existing_anchor_texts[anchor_text] for anchor_text in top_n_anchor_texts]

    max_similarity = max(similarities_filtered.values(), default=0)

    if top_n_data:
       return returnData(top_n_data,max_similarity)
    return returnData([],0)

def returnData(top_n_data,max_similarity):
    return {"data": top_n_data, "conversational_text": "", "similarities_level": max_similarity}


def find_most_similar_batch(input_text, top_n=2):
    return load_json(input_text, top_n, 0)
