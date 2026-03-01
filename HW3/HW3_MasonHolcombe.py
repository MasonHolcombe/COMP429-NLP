from collections import defaultdict
import string
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from HW3_Utilities import load_movie_polarity_reviews, tokenize, create_vocab_dict, create_review_vector, dataset_vectorizer    

def HW3_main():
    np.random.seed(11022001)

    all_reviews = load_movie_polarity_reviews()
    all_reviews_dict = dict(sorted(all_reviews.items(), key=lambda item: np.random.rand()))

    # 1)
    train_reviews = dict(list(all_reviews_dict.items())[:int(len(all_reviews_dict) * 0.7)])
    dev_reviews = dict(list(all_reviews_dict.items())[int(len(all_reviews_dict) * 0.7):int(len(all_reviews_dict) * 0.85)])
    test_reviews = dict(list(all_reviews_dict.items())[int(len(all_reviews_dict) * 0.85):])

    print(f"STEP 1 | Total: {len(all_reviews_dict)} | Train: {len(train_reviews)}, Dev: {len(dev_reviews)}, Test: {len(test_reviews)}")

    # 2)
    train_vocab = create_vocab_dict(train_reviews)

    n, m = 2, 800
    train_vocab_limited = {word: count for word, count in train_vocab.items() if count >= n and count < m}

    train_X, train_y = dataset_vectorizer(train_reviews, train_vocab_limited)
    dev_X, dev_y = dataset_vectorizer(dev_reviews, train_vocab_limited) # on limited train vocab
    test_X, test_y = dataset_vectorizer(test_reviews, train_vocab_limited) # on limited train vocab

    print(f"STEP 2 | Train & Dev Shape: {train_X.shape}, {dev_X.shape}")

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

    print(f"STEP 3 | Best Model Accuracy on Development Set: {best_acc:.2%}")

    # 4)
    test_best_lr_predictions = best_model.predict(test_X)
    test_accuracy = np.mean(test_y == test_best_lr_predictions)

    print(f"STEP 4 | Best Model Accuracy on Test Data (shape={test_X.shape}): {test_accuracy:.2%}")

    # 5)
    clean_train_reviews = [" ".join(tokenize(review)) for review in train_reviews.keys()]
    clean_dev_reviews = [" ".join(tokenize(review)) for review in dev_reviews.keys()]
    clean_test_reviews = [" ".join(tokenize(review)) for review in test_reviews.keys()]

    tfidf_vectorizer = TfidfVectorizer()
    train_X_tfidf = tfidf_vectorizer.fit_transform(clean_train_reviews)

    dev_X_tfidf = tfidf_vectorizer.transform(clean_dev_reviews) # just transform, fit using train vocab
    test_X_tfidf = tfidf_vectorizer.transform(clean_test_reviews) # just transform, fit using train vocab

    print(f"STEP 5.1 | TfIdf Train & Dev Shape: {train_X_tfidf.shape}, {dev_X_tfidf.shape}")

    tfidf_model_dev_scores = {}
    for C_val in param_grid['C']:
        
        lr_tfidf_model = LogisticRegression(**base_params, C=C_val)
        lr_tfidf_model.fit(train_X_tfidf, train_y)
        
        dev_tfidf_lr_predictions = lr_tfidf_model.predict(dev_X_tfidf)
        accuracy = np.mean(dev_y == dev_tfidf_lr_predictions)
        
        tfidf_model_dev_scores[f"lr_{C_val}"] = (lr_tfidf_model, accuracy)

    best_tfidf_param, (best_tfidf_model, best_tfidf_acc) = max(tfidf_model_dev_scores.items(), key=lambda x: x[1][1])

    print(f"STEP 5.2 | TfIdf Best Model Accuracy on Development Set: {best_tfidf_acc:.2%}")

    test_best_tfidf_lr_predictions = best_tfidf_model.predict(test_X_tfidf)
    test_tfidf_accuracy = np.mean(test_y == test_best_tfidf_lr_predictions)

    print(f"STEP 5.3 | TfIdf Best Model Accuracy on Test Data (shape={test_X_tfidf.shape}): {test_tfidf_accuracy:.2%}")

if __name__ == "__main__":
    HW3_main()