import gensim
import gensim.downloader as api

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(2026)


from HW5_Utilities import load_movie_polarity_reviews, pretrained_model_similarity, sentence_to_avg

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.metrics import accuracy_score


def main():
    # 1
    EMBEDDING_DIM = 300
    WINDOW_SIZE = 3

    dataset = api.load("text8")
    print(f"STEP 1 | Starting {EMBEDDING_DIM} dim training...")
    model = gensim.models.Word2Vec(sentences=dataset,
                                vector_size=EMBEDDING_DIM,
                                window=WINDOW_SIZE)

    print(f"STEP 1 | Completed {EMBEDDING_DIM} dim training.")

    similar_words = [
        ("king", "queen"),
        ("fast", "quick"),
        ("begin", "start"),
        ("ocean", "sea"),
        ("hot", "warm")
    ]

    print(f"STEP 1 | Similarity Scores:")
    for w1, w2 in similar_words:
        wv1 = model.wv[w1].reshape(1, -1)
        wv2 = model.wv[w2].reshape(1, -1)

        sim_score = cosine_similarity(wv1, wv2).item()

        print(f"\t({w1}, {w2}) -> {sim_score:.4f}")

    # 2
    print(f"STEP 2 | Downloading google-news-300...")
    pretrained_model = api.load('word2vec-google-news-300')
    print(f"STEP 2 | Download complete.")

    words = ["neural", "calculus", "happiness", "flower"]

    print(f"STEP 2 | Large model qualitative behavior:")
    for word in words:
        top5 = pretrained_model.most_similar(word, topn=5)
        
        print(f"\nTop 5 words similar to '{word}':")
        for i, (w, score) in enumerate(top5, start=1):
            print(f"{i}. {w:^18} {score:.4f}")

    # 3
    wordsim353 = pd.read_csv("wordsim_similarity_goldstandard.csv")
    wordsim353.columns = ["word1", "word2", "mean_human_score"]

    wordsim353["model_score"] = wordsim353.apply(lambda row: pretrained_model_similarity(pretrained_model, row["word1"], row["word2"]), axis=1)

    rank_coef = spearmanr(wordsim353["mean_human_score"], wordsim353["model_score"]).statistic.item()
    print(f"STEP 3 | Spearman's Rank Correlation Coefficient on WordSim-353: {rank_coef:.4f}")

    # 4
    # run - running + swimming = swim
    word_arthmetic1 = pretrained_model.most_similar(
        positive=["run", "swimming"],
        negative=["running"],
        topn=5
    )
    for i, (word, score) in enumerate(word_arthmetic1[:3], 1):
        print(f"STEP 4 | Top {i}: 'run' - 'running' + 'swimming' -> {word} (score: {score:.4f})")


    # hot - temperature + emotion = unbridled_passion, passion, excitment
    word_arthmetic2 = pretrained_model.most_similar(
        positive=["hot", "emotion"],
        negative=["temperature"],
        topn=5
    )
    for i, (word, score) in enumerate(word_arthmetic2[:3], 1):
        print(f"STEP 4 | Top {i}: 'hot' - 'temperature' + 'emotion' -> {word} (score: {score:.4f})")

    # einstein - physics + music = jonas_brothers, eminem, mariah_carey
    word_arthmetic3 = pretrained_model.most_similar(
        positive=["einstein", "music"],
        negative=["physics"],
        topn=5
    )
    for i, (word, score) in enumerate(word_arthmetic3[:3], 1):
        print(f"STEP 4 | Top {i}: 'einstein' - 'physics' + 'music' -> {word} (score: {score:.4f})")

    # 5
    embedding_dim = pretrained_model.vector_size
    all_reviews = load_movie_polarity_reviews()

    all_reviews_dict = dict(sorted(all_reviews.items(), key=lambda item: np.random.rand()))

    # Split into T/D/Test: 70/15/15
    train_reviews = dict(list(all_reviews_dict.items())[:int(len(all_reviews_dict) * 0.7)])
    dev_reviews = dict(list(all_reviews_dict.items())[int(len(all_reviews_dict) * 0.7):int(len(all_reviews_dict) * 0.85)])
    test_reviews = dict(list(all_reviews_dict.items())[int(len(all_reviews_dict) * 0.85):])

    train_X = np.array([sentence_to_avg(s, model=pretrained_model, embedding_dim=embedding_dim) for s in train_reviews])
    train_y = np.array(list(train_reviews.values()))

    dev_X = np.array([sentence_to_avg(s, model=pretrained_model, embedding_dim=embedding_dim) for s in dev_reviews])
    dev_y = np.array(list(dev_reviews.values()))

    test_X = np.array([sentence_to_avg(s, model=pretrained_model, embedding_dim=embedding_dim) for s in test_reviews])
    test_y = np.array(list(test_reviews.values()))

    input_dim = embedding_dim

    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='RMSprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_X, train_y,
        epochs=7,
        batch_size=8,
        validation_data=(dev_X, dev_y)
    )

    y_pred_probs = model.predict(test_X)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print(f"STEP 5 | Test accuracy: {accuracy_score(test_y, y_pred):.2%}")

if __name__ == "__main__":
    main()