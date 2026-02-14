import requests
import string
from collections import defaultdict
import numpy as np
import pandas as pd

def __main__():
    master_url = "https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity."
    pos_url = master_url + "pos"
    neg_url = master_url + "neg"

    resp_pos, resp_neg = requests.get(pos_url), requests.get(neg_url)

    text_pos = resp_pos.text.strip().split("\n")
    text_neg = resp_neg.text.strip().split("\n")

    pos_dict = {k: 1 for k in text_pos}
    neg_dict = {k: 0 for k in text_neg}

    all_reviews = pos_dict | neg_dict

    # 1)
    # Shuffle data in dict, with seed
    np.random.seed(2001)
    all_reviews_dict = dict(sorted(all_reviews.items(), key=lambda item: np.random.rand()))

    # Split into 70/15/15
    train_reviews = dict(list(all_reviews_dict.items())[:int(len(all_reviews_dict) * 0.7)])
    dev_reviews = dict(list(all_reviews_dict.items())[int(len(all_reviews_dict) * 0.7):int(len(all_reviews_dict) * 0.85)])
    test_reviews = dict(list(all_reviews_dict.items())[int(len(all_reviews_dict) * 0.85):])

    print(f"STEP 1 | Total: {len(all_reviews_dict)} | Train: {len(train_reviews)}, Dev: {len(dev_reviews)}, Test: {len(test_reviews)}")
    
    # 2)
    stop_words = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "com", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "that", "the", "this", "to", "was", "what", "when",
    "where", "who", "will", "with", "you"
    ]
    def tokenize(text):
        tokens = [x for x in text.split() if x not in string.punctuation and len(x) > 1]
        for punct in string.punctuation:
            tokens = [token.replace(punct, '') for token in tokens]
        return tokens
    class NaiveBayesClassifier:
        def __init__(self, train_data):
            self.train_reviews = train_data
            self.train_reviews_tokenized = [tokenize(review) for review in self.train_reviews.keys()]
            self.vocabulary = list(set(word for review in self.train_reviews_tokenized for word in review))
        
            self.all_train_pos_words = []
            self.all_train_neg_words = []
            for review, label in self.train_reviews.items():
                if label == 1: self.all_train_pos_words.extend(tokenize(review))
                elif label == 0: self.all_train_neg_words.extend(tokenize(review))

            self.pos_freq_dict = defaultdict(int)
            self.neg_freq_dict = defaultdict(int)
            for word in self.all_train_pos_words:
                self.pos_freq_dict[word] += 1
            for word in self.all_train_neg_words:
                self.neg_freq_dict[word] += 1

            for word in stop_words:
                if word in self.pos_freq_dict:
                    del self.pos_freq_dict[word]
                if word in self.neg_freq_dict:
                    del self.neg_freq_dict[word]

        def calculate_priors(self, reviews):
            positive_prior = sum(reviews.values()) / len(reviews)
            negative_prior = 1 - positive_prior
            return np.log(positive_prior), np.log(negative_prior)

        def calculate_conditional_probabilities(self, word, label, alpha=1):
            vocab_size = len(self.vocabulary)

            num = self.pos_freq_dict[word] + alpha if label == 1 else self.neg_freq_dict[word] + alpha
            denom = len(self.pos_freq_dict) + alpha * vocab_size if label == 1 else len(self.neg_freq_dict) + alpha * vocab_size
            
            return np.log(num / denom)

        def estimate_label_posterior(self, review, label, alpha):
            tokens = tokenize(review)
        
            pos_prior, neg_prior = self.calculate_priors(train_reviews)
            prior = pos_prior if label == 1 else neg_prior

            result = prior
            for token in tokens:
                result += self.calculate_conditional_probabilities(token, label, alpha=alpha)
        
            return result

        def naive_bayes_prediction(self, review, alpha):
            pos_posterior = self.estimate_label_posterior(review, 1, alpha=alpha)
            neg_posterior = self.estimate_label_posterior(review, 0, alpha=alpha)
            result = np.argmax([neg_posterior, pos_posterior])
            return result, (neg_posterior, pos_posterior)
            
        def prediction_accuracy(self, predictions, actuals):
            accuracy = sum(1 for pred, actual in zip(predictions, actuals) if pred == actual) / len(actuals)
            return accuracy

        def evaluate_on_dataset(self, reviews, alpha):
            predictions = []
            for review in reviews.keys():
                pred_label, (neg_post, pos_post) = self.naive_bayes_prediction(review, alpha=alpha)
                predictions.append((int(pred_label), neg_post, pos_post))
            actuals = list(reviews.values())
            accuracy = self.prediction_accuracy([pred for pred, _, _ in predictions], actuals)
            return accuracy
    alpha = 1
    naive_bayes_classifier = NaiveBayesClassifier(train_data = train_reviews)

    accuracy_on_train = naive_bayes_classifier.evaluate_on_dataset(train_reviews, alpha=alpha)
    accuracy_on_dev = naive_bayes_classifier.evaluate_on_dataset(dev_reviews, alpha=alpha)

    print(f"STEP 2 | Naive Bayes Accuracy on Train/Dev: {accuracy_on_train:.2%}/{accuracy_on_dev:.2%}")

    # 3) 
    naive_bayes_classifier_train_dev = NaiveBayesClassifier(train_data = train_reviews | dev_reviews)
    accuracy_on_train = naive_bayes_classifier_train_dev.evaluate_on_dataset(train_reviews | dev_reviews, alpha=alpha)
    accuracy_on_test = naive_bayes_classifier_train_dev.evaluate_on_dataset(test_reviews, alpha=alpha)
    
    print(f"STEP 3 | Naive Bayes Accuracy on Train & Dev/Test: {accuracy_on_train:.2%}/{accuracy_on_test:.2%}")

    # 4)
    posterior_list = []
    for test_review, test_label in test_reviews.items():
        pred_label, (neg_post, pos_post) = naive_bayes_classifier_train_dev.naive_bayes_prediction(test_review, alpha=alpha)
        posterior_list.append((test_review, test_label, pred_label, pos_post, neg_post))
    posterior_df = pd.DataFrame(posterior_list, columns=["review", "actual_label", "predicted_label", "pos_posterior", "neg_posterior"])
    
    # Certain
    posterior_df["diff"] = posterior_df["pos_posterior"] - posterior_df["neg_posterior"]
    certain_df = posterior_df.sort_values(by="diff", ascending=False) # (head/tail)
    # Uncertain
    posterior_df["abs_diff"] = posterior_df["diff"].abs()
    uncertain_df = posterior_df.sort_values(by="abs_diff", ascending=True)
    
    print("STEP 4 | Calculated certain/uncertain reviews based on posterior difference.")

    # 5)
    pos_words = list(naive_bayes_classifier_train_dev.pos_freq_dict.keys())
    all_pos_conditional_probs = {word: naive_bayes_classifier_train_dev.calculate_conditional_probabilities(word, 1, alpha=alpha) for word in pos_words}
    sorted_pos_conditional_probs = dict(sorted(all_pos_conditional_probs.items(), key=lambda item: item[1], reverse=True))

    neg_words = list(naive_bayes_classifier_train_dev.neg_freq_dict.keys())
    all_neg_conditional_probs = {word: naive_bayes_classifier_train_dev.calculate_conditional_probabilities(word, 0, alpha=alpha) for word in neg_words}
    sorted_neg_conditional_probs = dict(sorted(all_neg_conditional_probs.items(), key=lambda item: item[1], reverse=True))

    print("STEP 5 | Calculated top features for positive and negative classes.")


if __name__ == "__main__":
    __main__()