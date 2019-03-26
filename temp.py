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

