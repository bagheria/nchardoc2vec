import cgrams2vec
import gensim
import logging
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

if __name__ == '__main__':
    # data_file = os.getcwd() + "D:/Data/ohsumed-first-20000-docs/training/"
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

    # model_ted = gensim.models.FastText(
    #     t.documents,
    #     size=100,
    #     window=5,
    #     min_count=2,
    #     workers=10,
    #     sg=1)

#    model_ted.save("medic-fastText.model")

    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    # c2v = dict(zip(model_ted.wv.index2word, model_ted.wv.syn0))

    LR_w2v_mean_gensim = Pipeline([("word2vec vectorizer", cgrams2vec.MeanEmbeddingVectorizer(w2v)),
                                   ("LR", LogisticRegression(random_state=0,
                                                             solver='lbfgs',
                                                             multi_class='multinomial'))])


    # mult_nb = Pipeline(
    #     [("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    # bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    # mult_nb_tfidf = Pipeline(
    #     [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    # bern_nb_tfidf = Pipeline(
    #     [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    # # SVM - which is supposed to be more or less state of the art
    # # http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
    # svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
    # svc_tfidf = Pipeline(
    #     [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
    #
    # etree_w2v = Pipeline([("word2vec vectorizer", cgrams2vec.MeanEmbeddingVectorizer(w2v)),
    #                       ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    # etree_w2v_tfidf = Pipeline([("word2vec vectorizer", cgrams2vec.TfidfEmbeddingVectorizer(w2v)),
    #                             ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    # svc_w2v = Pipeline([("word2vec vectorizer", cgrams2vec.MeanEmbeddingVectorizer(w2v)),
    #                       ("extra trees", SVC(kernel="linear"))])
    # svc_w2v_tfidf = Pipeline([("word2vec vectorizer", cgrams2vec.TfidfEmbeddingVectorizer(w2v)),
    #                             ("extra trees", SVC(kernel="linear"))])
    # gp_w2v = Pipeline([("word2vec vectorizer", cgrams2vec.MeanEmbeddingVectorizer(w2v)),
    #                     ("extra trees", GaussianProcessClassifier(1.0 * RBF(1.0)))])
    # gp_w2v_tfidf = Pipeline([("word2vec vectorizer", cgrams2vec.TfidfEmbeddingVectorizer(w2v)),
    #                           ("extra trees", GaussianProcessClassifier(1.0 * RBF(1.0)))])
    # svm_sig_w2v = Pipeline([("word2vec vectorizer", cgrams2vec.MeanEmbeddingVectorizer(w2v)),
    #                     ("extra trees", SVC(kernel="sigmoid"))])
    # svm_sig_w2v_tfidf = Pipeline([("word2vec vectorizer", cgrams2vec.TfidfEmbeddingVectorizer(w2v)),
    #                           ("extra trees", SVC(kernel="sigmoid"))])
    # etree_c2v = Pipeline([("word2vec vectorizer", cgrams2vec.MeanEmbeddingVectorizer(c2v)),
    #                       ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    # etree_c2v_tfidf = Pipeline([("word2vec vectorizer", cgrams2vec.TfidfEmbeddingVectorizer(c2v)),
    #                             ("extra trees", ExtraTreesClassifier(n_estimators=200))])

    all_models = [
        # ("mult_nb", mult_nb),
        # ("mult_nb_tfidf", mult_nb_tfidf),
        # ("bern_nb", bern_nb),
        # ("bern_nb_tfidf", bern_nb_tfidf),
        # ("svc", svc),
        # ("svc_tfidf", svc_tfidf),
        ("LR_w2v_mean_gensim", LR_w2v_mean_gensim),
        # ("w2v_tree", etree_w2v),
        # ("w2v_tree_tfidf", etree_w2v_tfidf),
        # ("w2v_svc", svc_w2v),
        # ("w2v_svc_tfidf", svc_w2v_tfidf),
        # ("w2v_svm_sig", svm_sig_w2v),
        # ("w2v_svM_sig_tfidf", svm_sig_w2v_tfidf),
        # ("w2v_gp", gp_w2v),
        # ("w2v_gp_tfidf", gp_w2v_tfidf),
        # ("c2v", etree_c2v),
        # ("c2v_tfidf", etree_c2v_tfidf)
    ]

    for name, model in all_models:
        score = cross_val_score(model, t.documents, t.y_train, cv=5).mean()
        print(name, score)

    # scores = sorted([(name, cross_val_score(model, documents, y_train, cv=5).mean())
    #                  for name, model in all_models])
    # # key=lambda (_, x): -x)
    # print
    # tabulate(scores, floatfmt=".4f", headers=("model", 'score'))

    # # The two datasets must be the same size
    # max_dataset_size = len(model.wv.syn0)
    # print("len(documents) = ", len(documents))
    # print("max_dataset_size = ", max_dataset_size)
    # clf1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').\
    #     fit(model.wv.syn0, y_train[:max_dataset_size])
    #
    # # Prediction of the first 15 samples of all features
    # predict1 = clf1.predict(model.wv.syn0[:15, :])
    # predict2 = clf1.predict(model.wv.syn0[-35:, :])
    # # Calculating the score of the predictions
    # score = clf1.score(model.wv.syn0, y_train[:max_dataset_size])
    # print("\nPrediction fasttext : \n", predict1)
    # print("\nPrediction fasttext : \n", predict2)
    # print("Score word2vec : \n", score)
    #
    # max_dataset_size = len(model_ted.wv.syn0)
    # print("max_dataset_size = ", max_dataset_size)
    # clf2 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').\
    #     fit(model_ted.wv.syn0, y_train[:max_dataset_size])
    #
    # # Prediction of the first 15 samples of all features
    # predict1 = clf2.predict(model_ted.wv.syn0[:15, :])
    # predict2 = clf2.predict(model_ted.wv.syn0[-35:, :])
    # # Calculating the score of the predictions
    # score = clf2.score(model_ted.wv.syn0, y_train[:max_dataset_size])
    # print("\nPrediction fasttext : \n", predict1)
    # print("\nPrediction fasttext : \n", predict2)
    # print("Score fasttext : \n", score)
