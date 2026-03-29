import requests
import string
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def load_movie_polarity_reviews():
    master_url = "https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity."
    pos_url = master_url + "pos"
    neg_url = master_url + "neg"

    resp_pos, resp_neg = requests.get(pos_url), requests.get(neg_url)

    text_pos = resp_pos.text.strip().split("\n")
    text_neg = resp_neg.text.strip().split("\n")

    pos_dict = {k: 1 for k in text_pos}
    neg_dict = {k: 0 for k in text_neg}

    all_reviews = pos_dict | neg_dict

    return all_reviews

def pretrained_model_similarity(pretrained_model, word1, word2):
    wv1 = pretrained_model[word1].reshape(1, -1)
    wv2 = pretrained_model[word2].reshape(1, -1)

    sim_score = cosine_similarity(wv1, wv2).item()
    return sim_score

def sentence_to_avg(sentence, model, embedding_dim):
    tokens = sentence.lower().split()
    
    vectors = []
    
    for word in tokens:
        if word in model:
            vectors.append(model[word])
    
    if len(vectors) == 0:
        return np.zeros(embedding_dim)  # all OOV case
    
    return np.mean(vectors, axis=0)