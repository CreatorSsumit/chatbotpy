from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker
import json
# from conversational import chatbot
from apscheduler.schedulers.background import BackgroundScheduler
import time
import torch

# Load the SentenceTransformer model once
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Create a SpellChecker instance once
spell = SpellChecker()
scheduler = BackgroundScheduler()

json_file_path = "alldata.json"

# Load existing data from JSON once
try:
    with open(json_file_path, "r") as file:
        existing_data = json.load(file)
except FileNotFoundError:
    existing_data = []

# Preprocess existing data
existing_data = [
    {"anchor_text": data["anchor_text"], "anchor_link": data["anchor_link"]}
    for data in existing_data
    if data["anchor_text"].strip()
]
existing_anchor_texts = {
    ' '.join(data["anchor_text"].split("-")): data for data in existing_data}
existing_anchor_texts_set = set(existing_anchor_texts.keys())
existing_embeddings = {anchor_text: model.encode([anchor_text], convert_to_tensor=True)[
    0] for anchor_text in existing_anchor_texts}


def file_reader():
    global existing_data, existing_anchor_texts, existing_anchor_texts_set, existing_embeddings
    print('again')
    try:
        with open(json_file_path, "r") as file:
            existing_data_new = json.load(file)
    except FileNotFoundError:
        existing_data_new = []

    if len(existing_data) != len(existing_data_new):
        existing_data = existing_data_new

        existing_data = [
            {"anchor_text": data["anchor_text"],
             "anchor_link": data["anchor_link"]}
            for data in existing_data
        ]
        existing_data = [
            data for data in existing_data if data["anchor_text"].strip()]

        existing_anchor_texts = {
            ' '.join(data["anchor_text"].split("-")): data for data in existing_data}

        existing_anchor_texts_set = set(existing_anchor_texts.keys())

        existing_embeddings = {
            anchor_text: model.encode([anchor_text], convert_to_tensor=True)[0] for anchor_text in existing_anchor_texts
        }


def load_json(input_text, top_n, threshold):
    corrected_input_text = ' '.join(spell.correction(token) if spell.correction(token) else token for token in input_text.split())
    corrected_input_text = ' '.join(corrected_input_text.split("-"))

    input_embedding = model.encode([corrected_input_text], convert_to_tensor=True)[0]

    similarities = util.pytorch_cos_sim(input_embedding.unsqueeze(0), torch.stack(list(existing_embeddings.values())))
    similarities = similarities.squeeze().tolist()
    similarities_filtered = {anchor_text: score for anchor_text, score in zip(existing_embeddings.keys(), similarities) if score > threshold}

    top_n_anchor_texts = sorted(
        similarities_filtered, key=similarities_filtered.get, reverse=True)[:top_n]

    top_n_data = [existing_anchor_texts[anchor_text]
                  for anchor_text in top_n_anchor_texts]

    max_similarity = max(similarities_filtered.values(), default=0)

    return {"data": top_n_data, "conversational_text": "", "similarities_level": max_similarity}


def find_most_similar_batch(input_text, top_n=1):
    return load_json(input_text, top_n, 0)
