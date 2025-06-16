# entrainement

n_topics = 18

# 1. Séparation train/test
train_df, test_df = train_test_split(questionsclean1, test_size=0.2, random_state=42)

# 1. Tokenisation 
train_texts = (train_df['NewTitle'] + ' ' + train_df['NewBody']).str.lower().str.split()
test_texts = (test_df['NewTitle'] + ' ' + test_df['NewBody']).str.lower().str.split()

# 3. Création du dictionnaire uniquement sur les données d'entraînement
dictionary = Dictionary(train_texts)


# 4. Création des corpus (bag-of-words)
train_corpus = [dictionary.doc2bow(doc) for doc in train_texts]
test_corpus = [dictionary.doc2bow(doc) for doc in test_texts]

# 5. Entraînement du modèle LDA sur le corpus d'entraînement
lda_model_gensim2 = LdaModel(
    corpus=train_corpus,
    id2word=dictionary,
    num_topics=n_topics,
    random_state=42,
    update_every=1,
    chunksize=5000,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

if __name__ == "__main__":
    train()