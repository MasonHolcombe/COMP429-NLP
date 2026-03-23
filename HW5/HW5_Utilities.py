import requests
import string
from collections import defaultdict
import numpy as np

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

def tokenize_sentence(text):
    tokens = [x for x in text.split() if x not in string.punctuation and len(x) > 1]
    for punct in string.punctuation:
        tokens = [token.lower().replace(punct, '') for token in tokens]
    return tokens

def create_vocab_dict(reviews_dict):
    vocab = defaultdict(int)
    for r in reviews_dict.keys():
        tokens = tokenize_sentence(r)
        for t in tokens:
            vocab[t] += 1
    return dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))

def create_review_vector(review, vocab_dict):
    vector = np.zeros(len(vocab_dict) + 1)
    tokens = tokenize_sentence(review)
    for i, word in enumerate(vocab_dict.keys()):
        if word in tokens:
            vector[i] = 1

    oov_tokens = list(set([token for token in tokens if token not in vocab_dict]))
    if oov_tokens:
        vector[-1] = len(oov_tokens)

    return vector

def dataset_vectorizer(reviews, vocab_dict):
    X, y = [], []
    for review, label in reviews.items():
        X.append(create_review_vector(review, vocab_dict))
        y.append(label)
    return np.array(X), np.array(y)