import requests
import string
from collections import defaultdict
import numpy as np

def main():
    # Helper Functions
    def tokenizer(text):
        tokens = [x for x in text.split() if x not in string.punctuation and len(x) > 1]
        for punct in string.punctuation:
            tokens = [token.replace(punct, '') for token in tokens]
        return tokens
    def count_word_frequencies(review_tokens, filler_word_threshold=500):
        word_freq = defaultdict(int)
        for token in review_tokens:
            pos_freq = pos_word_freq[token]
            neg_freq = neg_word_freq[token]
            
            combined = pos_freq + neg_freq
            word_freq[token] = (pos_freq, neg_freq,
                                pos_freq / combined if combined > 0 else 0,
                                neg_freq / combined if combined > 0 else 0)

        word_freq = {word: freqs for word, freqs in word_freq.items() if freqs[0] <= filler_word_threshold and freqs[1] <= filler_word_threshold}
        return word_freq
    def compute_prediction(word_freq):
        avgs = np.mean([[freqs[2], freqs[3]] for freqs in word_freq.values()], axis=0)
        return avgs, np.argmax(avgs)
    def classifier(review):
        tokens = tokenizer(review)
        word_freq = count_word_frequencies(tokens)
        avgs, prediction = compute_prediction(word_freq)
        sentiment = "Positive" if prediction == 0 else "Negative"
        return sentiment
    
    # Data Loading
    master_url = "https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity."
    pos_url = master_url + "pos"
    neg_url = master_url + "neg"

    resp_pos, resp_neg = requests.get(pos_url), requests.get(neg_url)

    # Data Processing
    text_pos = resp_pos.text.strip().split("\n")
    text_neg = resp_neg.text.strip().split("\n")

    all_pos_text_list = ' '.join([review for review in text_pos]).split()
    all_neg_text_list = ' '.join([review for review in text_neg]).split()

    all_pos_words = [tokenizer(word)[0] for word in all_pos_text_list if len(tokenizer(word)) > 0]
    all_neg_words = [tokenizer(word)[0] for word in all_neg_text_list if len(tokenizer(word)) > 0]

    pos_reviews = {text_pos[i]: "Positive" for i in range(len(text_pos))}
    neg_reviews = {text_neg[i]: "Negative" for i in range(len(text_neg))}
    all_reviews = pos_reviews | neg_reviews

    # Build Frequency Tables
    pos_word_freq = defaultdict(int)
    for pos_token in all_pos_words:
        pos_word_freq[pos_token] += 1
        
    neg_word_freq = defaultdict(int)
    for neg_token in all_neg_words:
        neg_word_freq[neg_token] += 1

    # Prediction
    predictions = {review: classifier(review) for review in all_reviews.keys()}

    # Evaluation
    correct = sum(1 for review, true_label in all_reviews.items() if predictions[review] == true_label)
    accuracy = correct / len(all_reviews)

    print(f"Accuracy of Average +/- Frequency Classifier: {accuracy:.2%} ({correct} / {len(all_reviews)})")


if __name__ == "__main__":
    main()