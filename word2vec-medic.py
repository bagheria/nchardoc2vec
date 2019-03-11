import gensim
import logging
import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


# Need to finish the next function
# def torch_word2ve():
# using torch to create character based n-gram word vectors


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def show_file_contents(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


def re_tokenize(text):
    """ Tokenize input test using defined regular expression
    :param text: Input text
    :type text: str
    :return: Tokens
    :rtype: str
    """
    re_pattern = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
    for match in re_pattern.finditer(text):
        yield match.group()


def tokenize(text, deacc=False, encoding='utf8', lowercase=False, to_lower=False):
    """ Iteratively yield re-based tokens as unicode strings, removing accent marks and optionally lowercasing
    :param text: Input text
    :type text: str
    :param deacc: Remove accentuation
    :type deacc: bool
    :param encoding: Encoding of text
    :type encoding: str
    :param lowercase: To lowercase
    :type lowercase: bool
    :param to_lower: To lowercase
    :type to_lower: bool
    :return: Contiguous sequences of alphabetic characters (no digits!)
    :rtype: str
    # Example:
    # list(tokenize('Nic nemůže letět rychlostí vyšší, než 300 tisíc kilometrů za sekundu!', deacc=True))
    # [u'Nic', u'nemuze', u'letet', u'rychlosti', u'vyssi', u'nez', u'tisic', u'kilometru', u'za', u'sekundu']
    """
    lowercase = lowercase or to_lower
    from gensim.utils import to_unicode
    text = to_unicode(text, encoding, errors='ignore')
    if lowercase:
        text = text.lower()
    if deacc:
        from gensim.utils import deaccent
        # Example
        # --------
        # >>> from gensim.utils import deaccent
        # >>> deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
        # u'Sef chomutovskych komunistu dostal postou bily prasek'
        text = deaccent(text)
    return re_tokenize(text)


def preprocess(doc, deacc=False, min_len=2, max_len=15):
    """ Convert input text into a list of tokens ( regex, lowercase, etc.)
    :param doc: Input text
    :type doc: str
    :param deacc: Remove accentuation
    :type deacc: bool
    :param min_len:  Minimal length of token
    :type min_len: int
    :param max_len: Maximal length of token
    :type max_len: int
    :return: Tokens
    :rtype: str
    """
    tokens = [
        token for token in tokenize(doc, to_lower=True, deacc=deacc)
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


def read_input(train_path):
    categories = os.listdir(train_path)
    s = 0
    for i in range(categories.__len__()):
        data_path = train_path + str(categories[i]) + "/"
        for j, input_file in enumerate(os.listdir(data_path)):
            if j % 100 == 0:
                s = s + 100
                logging.info("read {0} sentences".format(s))
            with open(data_path + input_file, 'rb') as f:
                for line in f:
                    # do some pre-processing and return list of words for each text
                    yield preprocess(line)


def read_target(train_path):
    categories = os.listdir(train_path)
    Y_dataset = []
    for i in range(categories.__len__()):
        data_path = train_path + str(categories[i]) + "/"
        for j, input_file in enumerate(os.listdir(data_path)):
            Y_dataset.append(i)
    return Y_dataset


class MeanEmbeddingVectorizer(object):
    """
    Averaging word vectors for all words in a text
    """
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = 150
        # self.dim = len(next(word2vec.values()))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = 150
        # self.dim = len(word2vec.values().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


if __name__ == '__main__':
    data_file = os.getcwd() + "/Sentence classification/"

    # read the tokenized texts into a list
    # each text item becomes a series of words
    # so this becomes a list of lists
    y_train = read_target(data_file)
    documents = list(read_input(data_file))
    logging.info("Done reading data file")

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        size=150,
        window=10,
        min_count=2,
        workers=10)
    # Default algo in Gensim is CBOW. It is used if you simply call gensim.models.Word2Vec(sentences)
    # In order to use skipgram run gensim.models.Word2Vec(sentences, sg=1)
    model.train(documents, total_examples=len(documents), epochs=10)
#    model.save("medic-word2vec.model")

    model_ted = gensim.models.FastText(
        documents,
        size=100,
        window=5,
        min_count=2,
        workers=10,
        sg=1)

#    model_ted.save("medic-fastText.model")

    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    c2v = dict(zip(model_ted.wv.index2word, model_ted.wv.syn0))

    mult_nb = Pipeline(
        [("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    mult_nb_tfidf = Pipeline(
        [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    bern_nb_tfidf = Pipeline(
        [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    # SVM - which is supposed to be more or less state of the art
    # http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
    svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
    svc_tfidf = Pipeline(
        [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])

    etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                          ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    svc_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                          ("extra trees", SVC(kernel="linear"))])
    svc_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                ("extra trees", SVC(kernel="linear"))])
    gp_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                        ("extra trees", GaussianProcessClassifier(1.0 * RBF(1.0)))])
    gp_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                              ("extra trees", GaussianProcessClassifier(1.0 * RBF(1.0)))])
    svm_sig_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                        ("extra trees", SVC(kernel="sigmoid"))])
    svm_sig_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                              ("extra trees", SVC(kernel="sigmoid"))])
    etree_c2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(c2v)),
                          ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    etree_c2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(c2v)),
                                ("extra trees", ExtraTreesClassifier(n_estimators=200))])

    all_models = [
        # ("mult_nb", mult_nb),
        # ("mult_nb_tfidf", mult_nb_tfidf),
        # ("bern_nb", bern_nb),
        # ("bern_nb_tfidf", bern_nb_tfidf),
        # ("svc", svc),
        # ("svc_tfidf", svc_tfidf),
        ("w2v_tree", etree_w2v),
        ("w2v_tree_tfidf", etree_w2v_tfidf),
        ("w2v_svc", svc_w2v),
        ("w2v_svc_tfidf", svc_w2v_tfidf),
        ("w2v_svm_sig", svm_sig_w2v),
        ("w2v_svM_sig_tfidf", svm_sig_w2v_tfidf),
        # ("w2v_gp", gp_w2v),
        # ("w2v_gp_tfidf", gp_w2v_tfidf),
        # ("c2v", etree_c2v),
        # ("c2v_tfidf", etree_c2v_tfidf)
    ]

    for name, model in all_models:
        score = cross_val_score(model, documents, y_train, cv=5).mean()
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
