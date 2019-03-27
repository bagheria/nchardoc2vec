import cgrams2vec
import gensim
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
from doc2vec_gensim import Doc2Vec, TaggedDocument

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

if __name__ == '__main__':
    data_file = "D:/Data/ohsumed-first-20000-docs/training/"

    # read the tokenized texts into a list
    # each text item becomes a series of words
    # so this becomes a list of lists
    t = cgrams2vec.cgrams2vec(data_file, num_epochs=2)
    # t.create_dataset()
    # t.train_embedding()
    # y_train = word2vec_medic.read_target(data_file)
    # documents = list(word2vec_medic.read_input(data_file))
    logging.info("Done reading data file")
    # t.train_skipgram()

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        t.documents,
        size=150,
        window=10,
        min_count=2,
        workers=10)
    # Default algo in Gensim is CBOW. It is used if you simply call gensim.models.Word2Vec(sentences)
    # In order to use skipgram run gensim.models.Word2Vec(sentences, sg=1)
    model.train(t.documents, total_examples=len(t.documents), epochs=10)
#    model.save("medic-word2vec.model")

    model_ted = gensim.models.FastText(
        t.documents,
        size=100,
        window=5,
        min_count=2,
        workers=10,
        sg=1)

#    model_ted.save("medic-fastText.model")

    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    c2v = dict(zip(model_ted.wv.index2word, model_ted.wv.syn0))

    LR_w2v_mean = Pipeline([("word2vec vectorizer", cgrams2vec.MeanEmbeddingVectorizer(w2v)),
                                   ("LR", LogisticRegression(random_state=0,
                                                             solver='lbfgs',
                                                             max_iter=100,
                                                             multi_class='multinomial'))])
    LR_w2v_tfidf = Pipeline([("word2vec vectorizer", cgrams2vec.TfidfEmbeddingVectorizer(w2v)),
                                   ("LR", LogisticRegression(random_state=0,
                                                             solver='lbfgs',
                                                             max_iter=100,
                                                             multi_class='multinomial'))])

    LR_c2v_mean = Pipeline([("char2vec vectorizer", cgrams2vec.MeanEmbeddingVectorizer(c2v)),
                                   ("LR", LogisticRegression(random_state=0,
                                                             solver='lbfgs',
                                                             max_iter=100,
                                                             multi_class='multinomial'))])
    LR_c2v_tfidf = Pipeline([("char2vec vectorizer", cgrams2vec.TfidfEmbeddingVectorizer(c2v)),
                                    ("LR", LogisticRegression(random_state=0,
                                                              solver='lbfgs',
                                                              max_iter=100,
                                                              multi_class='multinomial'))])

    # doc2vec model from gensim
    tagged_data = [TaggedDocument(words=t.documents[i], tags=[str(i)]) for i, _d in enumerate(t.documents)]
    max_epochs = 5 # 100
    vec_size = 20
    alpha = 0.025
    d_model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)
    d_model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        d_model.train(tagged_data,
                    total_examples=d_model.corpus_count,
                    epochs=d_model.iter)
        # decrease the learning rate
        d_model.alpha -= 0.0002
        # fix the learning rate, no decay
        d_model.min_alpha = d_model.alpha

    LR_doc2vec = Pipeline([("doc2vec vectorizer", d_model.docvecs),
                                  ("LR", LogisticRegression(random_state=0,
                                                            solver='lbfgs',
                                                            max_iter=100,
                                                            multi_class='multinomial'))])

    # run all the models with logistic regression:
    all_models = [
        ("LR_w2v_mean", LR_w2v_mean),
        ("LR_w2v_tfidf", LR_w2v_tfidf),
        ("LR_c2v_mean", LR_c2v_mean),
        ("LR_c2v_tfidf", LR_c2v_tfidf),
        ("LR_doc2vec", LR_doc2vec)
    ]

    for name, model in all_models:
        score = cross_val_score(model, t.documents, t.y_train, cv=5).mean()
        print(name, score)
