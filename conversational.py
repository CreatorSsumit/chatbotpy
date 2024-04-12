import re
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import random
import numpy as np
import nltk
from typing import List

nltk.download('punkt')
nltk.download('wordnet')

def chatbot(input_text):
    with open('allData.txt', 'r', encoding='utf-8') as file:
        file_content = file.read()
    
    data_text = file_content.lower()
    data_text = re.sub(r'\[[0-9]*\]', ' ', data_text)
    data_text = re.sub(r'\s+', ' ', data_text)
    
    sen = nltk.sent_tokenize(data_text)
    wnlem = nltk.stem.WordNetLemmatizer()

    def performing_lemmatization(tokens):
        return [wnlem.lemmatize(token) for token in tokens]

    pr = dict((ord(punctuation), None) for punctuation in string.punctuation)

    def get_processed_text(document):
        return performing_lemmatization(nltk.word_tokenize(document.lower().translate(pr)))

    greet_inputs = ("hey", "hello", "good morning", "good afternoon",
                    "good evening", "good night", "hii", "what's up")
    greeting_response = ("hey", "how are you?", "how are you doing",
                        "i am good and you", "welcome", "good to see you back")

    def generate_greet_response(greeting):
        for token in greeting.split():
            if token.lower() in greet_inputs:
                return random.choice(greeting_response)

    def generate_response(user_input):
        bot_response = ''
        sen.append(user_input)

        word_vectorizer = TfidfVectorizer(
            tokenizer=get_processed_text, stop_words='english')
        word_vectors = word_vectorizer.fit_transform(sen)
        similar_vector_values = cosine_similarity(word_vectors[-1], word_vectors)
        similar_sentence_number = similar_vector_values.argsort()[0][-2]
        matched_vector = similar_vector_values.flatten()
        matched_vector.sort()
        vector_matched = matched_vector[-2]

        if vector_matched == 0:
            bot_response = bot_response + \
                "I am sorry I don't understand, can you briefly explain it to me"
        else:
            bot_response = bot_response + sen[similar_sentence_number]
            return bot_response

    warnings.filterwarnings("ignore")
    nltk.download('wordnet')

    
    human = input_text
    human = human.lower()
    
    if generate_greet_response(human) is not None:
        return generate_greet_response(human)
    else:
        return generate_response(human)





