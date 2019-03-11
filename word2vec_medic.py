import logging
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


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
