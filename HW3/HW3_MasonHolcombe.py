import requests
from collections import defaultdict
import string
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from HW3_Utilities import tokenize, create_vocab_dict, create_review_vector, dataset_vectorizer    

def main():
    master_url = "https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity."
    pos_url = master_url + "pos"
    neg_url = master_url + "neg"

    resp_pos, resp_neg = requests.get(pos_url), requests.get(neg_url)

    text_pos = resp_pos.text.strip().split("\n")
    text_neg = resp_neg.text.strip().split("\n")

    pos_dict = {k: 1 for k in text_pos}
    neg_dict = {k: 0 for k in text_neg}

    all_reviews = pos_dict | neg_dict

    # Shuffle data in dict, with seed
    np.random.seed(2026)
    all_reviews_dict = dict(sorted(all_reviews.items(), key=lambda item: np.random.rand()))

    # 1)

    # Split into T/D/Test: 70/15/15
    train_reviews = dict(list(all_reviews_dict.items())[:int(len(all_reviews_dict) * 0.7)])
    dev_reviews = dict(list(all_reviews_dict.items())[int(len(all_reviews_dict) * 0.7):int(len(all_reviews_dict) * 0.85)])
    test_reviews = dict(list(all_reviews_dict.items())[int(len(all_reviews_dict) * 0.85):])

    # in format all data: train: 70%, dev: 15%, test: 15%
    print(f"Total: {len(all_reviews_dict)} | Train: {len(train_reviews)}, Dev: {len(dev_reviews)}, Test: {len(test_reviews)}")

    # 2)

    train_vocab = create_vocab_dict(train_reviews)

    n, m = 2, 800
    train_vocab_limited = {word: count for word, count in train_vocab.items() if count >= n and count < m}

    train_X, train_y = dataset_vectorizer(train_reviews, train_vocab_limited)
    dev_X, dev_y = dataset_vectorizer(dev_reviews, train_vocab_limited) # on limited train vocab
    test_X, test_y = dataset_vectorizer(test_reviews, train_vocab_limited) # on limited train vocab

    print(f"Train X shape: {train_X.shape}, {train_y.shape}")
    print(f"Dev X shape: {dev_X.shape}, {dev_y.shape}")

    # 3) 

    base_params = {
        'random_state': 0,
        'max_iter': 1000,
        }
    param_grid = {
        'C': [0.01, 0.1, 0.5, 1, 10, 100],
    }

    model_dev_scores = {}
    for C_val in param_grid['C']:
        
        lr_model = LogisticRegression(**base_params, C=C_val)
        lr_model.fit(train_X, train_y)
        
        dev_lr_predictions = lr_model.predict(dev_X)
        accuracy = np.mean(dev_y == dev_lr_predictions)
        
        model_dev_scores[f"lr_{C_val}"] = (lr_model, accuracy)

    best_param, (best_model, best_acc) = max(model_dev_scores.items(), key=lambda x: x[1][1])
    print(f"Best Development Set Accuracy: {best_acc:.2%}\n{best_model}")

    # 4)

    print(f"Test X shape: {test_X.shape}, {test_y.shape}")

    test_best_lr_predictions = best_model.predict(test_X)
    test_accuracy = np.mean(test_y == test_best_lr_predictions)
    print(f"Test Set Accuracy with Best Model: {test_accuracy:.2%}")

    # 5)

    clean_train_reviews = [" ".join(tokenize(review)) for review in train_reviews.keys()]
    clean_dev_reviews = [" ".join(tokenize(review)) for review in dev_reviews.keys()]
    clean_test_reviews = [" ".join(tokenize(review)) for review in test_reviews.keys()]

    tfidf_vectorizer = TfidfVectorizer()
    train_X_tfidf = tfidf_vectorizer.fit_transform(clean_train_reviews)

    dev_X_tfidf = tfidf_vectorizer.transform(clean_dev_reviews) # just transform, fit using train vocab
    test_X_tfidf = tfidf_vectorizer.transform(clean_test_reviews) # just transform, fit using train vocab

    tfidf_model_dev_scores = {}
    for C_val in param_grid['C']:
        
        lr_tfidf_model = LogisticRegression(**base_params, C=C_val)
        lr_tfidf_model.fit(train_X_tfidf, train_y)
        
        dev_tfidf_lr_predictions = lr_tfidf_model.predict(dev_X_tfidf)
        accuracy = np.mean(dev_y == dev_tfidf_lr_predictions)
        
        tfidf_model_dev_scores[f"lr_{C_val}"] = (lr_tfidf_model, accuracy)

    best_tfidf_param, (best_tfidf_model, best_tfidf_acc) = max(tfidf_model_dev_scores.items(), key=lambda x: x[1][1])
    print(f"Best Development Set Accuracy: {best_tfidf_acc:.2%}\n{best_tfidf_model}")

    print(f"Test X shape: {test_X_tfidf.shape}, {test_y.shape}")

    test_best_tfidf_lr_predictions = best_tfidf_model.predict(test_X_tfidf)
    test_tfidf_accuracy = np.mean(test_y == test_best_tfidf_lr_predictions)
    print(f"Test Set Accuracy with Best Model: {test_tfidf_accuracy:.2%}")

if __name__ == "__main__":
    main()