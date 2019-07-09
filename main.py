import cgrams2vec
import gensim
import logging
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
# from doc2vec_gensim import Doc2Vec, TaggedDocument
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.multiclass import OneVsRestClassifier


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def vector_for_learning(model, tagged_docs):
    regressors = [(model.infer_vector(doc.words, steps=20)) for doc in tagged_docs]
    return regressors


if __name__ == '__main__':
    data_file = "D:/Data/ohsumed-first-20000-docs/training/"

    # read the tokenized texts into a list     # each text item becomes a series of words
    # so this becomes a list of lists
    t = cgrams2vec.cgrams2vec(data_file, num_epochs=2)

    # t.create_dataset()     # t.train_embedding()     # y_train = word2vec_medic.read_target(data_file)
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

    model_ted.wv.most_similar("hart")

#    model_ted.save("medic-fastText.model")

    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    c2v = dict(zip(model_ted.wv.index2word, model_ted.wv.syn0))

    # pipeline #1
    LR_w2v_mean = Pipeline([("word2vec vectorizer", cgrams2vec.MeanEmbeddingVectorizer(w2v)),
                                   ("LR", OneVsRestClassifier(LogisticRegression(random_state=0,
                                                                                 solver='lbfgs',
                                                                                 max_iter=100,
                                                                                 multi_class='multinomial')))])
    # pipeline #2
    LR_w2v_tfidf = Pipeline([("word2vec vectorizer", cgrams2vec.TfidfEmbeddingVectorizer(w2v)),
                                   ("LR", OneVsRestClassifier(LogisticRegression(random_state=0,
                                                                                 solver='lbfgs',
                                                                                 max_iter=100,
                                                                                 multi_class='multinomial')))])
    # pipeline #3
    LR_c2v_mean = Pipeline([("char2vec vectorizer", cgrams2vec.MeanEmbeddingVectorizer(c2v)),
                                   ("LR", OneVsRestClassifier(LogisticRegression(random_state=0,
                                                                                 solver='lbfgs',
                                                                                 max_iter=100,
                                                                                 multi_class='multinomial')))])
    # pipeline #4
    LR_c2v_tfidf = Pipeline([("char2vec vectorizer", cgrams2vec.TfidfEmbeddingVectorizer(c2v)),
                                    ("LR", OneVsRestClassifier(LogisticRegression(random_state=0,
                                                                                  solver='lbfgs',
                                                                                  max_iter=100,
                                                                                  multi_class='multinomial')))])

    # pipeline #5: doc2vec model
    dc = 1 # 1 it you want to run doc2vec model
    if dc == 1:
        tagged_data_train = [TaggedDocument(words=t.documents[i],
                                            tags=[str(i)])
                             for i, _d in enumerate(t.documents)]
        max_epochs = 50  # 100
        vec_size = 20
        alpha = 0.025
        d_model = Doc2Vec(size=vec_size,
                          alpha=alpha,
                          min_alpha=0.00025,
                          min_count=1,
                          dm=1)
        d_model.build_vocab(tagged_data_train)
        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            d_model.train(tagged_data_train,
                          total_examples=d_model.corpus_count,
                          epochs=d_model.iter)
            # decrease the learning rate
            d_model.alpha -= 0.0002
            # fix the learning rate, no decay
            d_model.min_alpha = d_model.alpha

        # run doc2vec model
        X_train_doc2vec = vector_for_learning(d_model, tagged_data_train)
        # logreg = LogisticRegression(n_jobs=1, C=1e5)
        # print("LR_doc2vec ", cross_val_score(logreg, X_train_doc2vec, t.y_train, cv=5).mean())

    pipeline5_logreg = OneVsRestClassifier(LogisticRegression(random_state=0,
                                                              solver='lbfgs',
                                                              max_iter=100,
                                                              multi_class='multinomial'))

    # ---------------------- RUN ALL THE PIPELINES ----------------------
    # pipeline #1
    LR_w2v_mean.fit(t.documents,
                    t.y_train,
                    cv=5)
    # pipeline #2
    LR_w2v_tfidf.fit(t.documents,
                     t.y_train,
                     cv=5)
    # pipeline #3
    LR_c2v_mean.fit(t.documents,
                    t.y_train,
                    cv=5)
    # pipeline #4
    LR_c2v_tfidf.fit(t.documents,
                     t.y_train,
                     cv=5)
    # pipeline #5
    pipeline5_logreg.fit(X_train_doc2vec, y_train)

    # # run word2vec models with logistic regression:
    # all_models = [
    #     ("LR_w2v_mean", LR_w2v_mean),
    #     ("LR_w2v_tfidf", LR_w2v_tfidf),
    #     ("LR_c2v_mean", LR_c2v_mean),
    #     ("LR_c2v_tfidf", LR_c2v_tfidf),
    # ]
    #
    # for name, model in all_models:
    #     score = cross_val_score(model, t.documents, t.y_train, cv=5).mean()
    #     print(name, score)
    #
    # # run lr model for count vectors
    # docs = []
    # for i in range(t.documents.__len__()):
    #     docs.append((" ".join(t.documents[i])))
    # vectorizer = CountVectorizer()
    # X_count = vectorizer.fit_transform(docs).toarray()
    # print("LR_countvectors ", cross_val_score(logreg, X_count, t.y_train, cv=5).mean())
    #
    # # run lr model for tfidf vectors
    # tfidfconverter = TfidfVectorizer()
    # X_tfidf = tfidfconverter.fit_transform(docs).toarray()
    # print("LR_tfidfvectors ", cross_val_score(logreg, X_tfidf, t.y_train, cv=5).mean())
    #
    # # from sklearn.multiclass import OneVsRestClassifier
    # # from sklearn.svm import SVC
    # # classif = OneVsRestClassifier(SVC(kernel='linear'))
    # # print("OneVsRestClassifier_SVC_countvectors ", cross_val_score(classif, X_count, t.y_train, cv=5).mean())
    # # print("OneVsRestClassifier_SVC_tfidfvectors ", cross_val_score(classif, X_tfidf, t.y_train, cv=5).mean())
