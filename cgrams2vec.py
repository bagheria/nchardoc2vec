import logging
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)


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
                logging.info("read {0} documents".format(s))
            with open(data_path + input_file, 'rb') as f:
                # for line in f:
                # do some pre-processing and return list of words for each text
                yield preprocess(f.read())


def read_target(train_path):
    categories = os.listdir(train_path)
    Y_dataset = []
    for i in range(categories.__len__()):
        data_path = train_path + str(categories[i]) + "/"
        for j, input_file in enumerate(os.listdir(data_path)):
            Y_dataset.append(i)
    return Y_dataset

def read_target(file):
    categories = os.listdir(train_path)
    Y_dataset = []
    for i in range(categories.__len__()):
        data_path = train_path + str(categories[i]) + "/"
        for j, input_file in enumerate(os.listdir(data_path)):
            Y_dataset.append(i)
    return Y_dataset


class cgrams2vec:
    def __init__(self, data_file, embedding_dims=100, num_epochs=5, learning_rate=0.001):
        self.context_size = 2  # 2 words to the left, 2 to the right
        self.y_train = read_target(data_file)
        self.documents = list(read_input(data_file))
        self.vocabulary = self.get_vocab()
        self.word2idx = {w: idx for (idx, w) in enumerate(self.vocabulary)}
        self.idx2word = {idx: w for (idx, w) in enumerate(self.vocabulary)}
        self.embd_size = embedding_dims
        self.learning_rate = learning_rate
        self.n_epoch = num_epochs
        self.vocab_size = len(self.vocabulary)
        self.losses = []
        self.model = None
        self.idx_pairs = []
        self.W1 = None
        self.W2 = None

    def get_vocab(self):
        vocab = []
        for sentence in self.documents:
            for token in sentence:
                if token not in vocab:
                    vocab.append(token)
        return vocab

    def create_cbow_dataset(self):
        data = []
        for sentence in self.documents:
            # change to using CONTEXT_SIZE, use create_dataset function
            for i in range(2, len(sentence) - 2):
                context = [sentence[i - 2], sentence[i - 1],
                           sentence[i + 1], sentence[i + 2]]
                target = sentence[i]
                data.append((context, target))
        return data

    def create_skipgram_dataset(self):
        import random
        data = []
        for sentence in self.documents:
            # change to using CONTEXT_SIZE
            for i in range(2, len(sentence) - 2):
                data.append((sentence[i], sentence[i-2], 1))
                data.append((sentence[i], sentence[i-1], 1))
                data.append((sentence[i], sentence[i+1], 1))
                data.append((sentence[i], sentence[i+2], 1))
                # negative sampling
                for _ in range(4):
                    if random.random() < 0.5 or i >= len(sentence) - 3:
                        rand_id = random.randint(0, i-1)
                    else:
                        rand_id = random.randint(i+3, len(sentence)-1)
                    data.append((sentence[i], sentence[rand_id], 0))
        return data

    def train_cbow(self):
        hidden_size = 64
        loss_fn = nn.NLLLoss()
        self.model = CBOW(self.vocab_size, self.embd_size, self.context_size, hidden_size)
        # print(model)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        cbow_train = self.create_cbow_dataset()

        for epoch in range(self.n_epoch):
            total_loss = .0
            for context, target in cbow_train:
                ctx_idxs = [self.word2idx[w] for w in context]
                ctx_var = Variable(torch.LongTensor(ctx_idxs))

                self.model.zero_grad()
                log_probs = self.model(ctx_var)

                loss = loss_fn(log_probs, Variable(torch.LongTensor([self.word2idx[target]])))

                loss.backward()
                optimizer.step()

                total_loss += loss.data  # [0]
            self.losses.append(total_loss)

    def train_skipgram(self):
        loss_fn = nn.MSELoss()
        self.model = SkipGram(self.vocab_size, self.embd_size)
        # print(model)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        skipgram_train = self.create_skipgram_dataset()

        for epoch in range(self.n_epoch):
            total_loss = .0
            for in_w, out_w, target in skipgram_train:
                in_w_var = Variable(torch.LongTensor([self.word2idx[in_w]]))
                out_w_var = Variable(torch.LongTensor([self.word2idx[out_w]]))

                self.model.zero_grad()
                log_probs = self.model(in_w_var, out_w_var)
                loss = loss_fn(log_probs[0], Variable(torch.Tensor([target])))

                loss.backward()
                optimizer.step()

                total_loss += loss.data  # [0]
            self.losses.append(total_loss)

    def create_dataset(self):
        # for each sentence
        for sentence in self.documents:
            indices = [self.word2idx[word] for word in sentence]
            # for each word, treated as center word
            for center_word_pos in range(len(indices)):
                # for each window position
                for w in range(-self.context_size, self.context_size + 1):
                    context_word_pos = center_word_pos + w
                    # make sure not jump out sentence
                    if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                        continue
                    context_word_idx = indices[context_word_pos]
                    self.idx_pairs.append((indices[center_word_pos], context_word_idx))

        self.idx_pairs = np.array(self.idx_pairs)  # it will be useful to have this as numpy array

    def get_input_layer(self, word_idx):
        x = torch.zeros(self.vocab_size).float()
        x[word_idx] = 1.0
        return x

    def train_embedding(self):
        self.W1 = Variable(torch.randn(self.embd_size, self.vocab_size).float(), requires_grad=True)
        self.W2 = Variable(torch.randn(self.vocab_size, self.embd_size).float(), requires_grad=True)

        for epo in range(self.n_epoch):
            loss_val = 0
            for data, target in self.idx_pairs:
                x = Variable(self.get_input_layer(data)).float()
                y_true = Variable(torch.from_numpy(np.array([target])).long())

                z1 = torch.matmul(self.W1, x)
                z2 = torch.matmul(self.W2, z1)

                log_softmax = F.log_softmax(z2, dim=0)

                loss = F.nll_loss(log_softmax.view(1, -1), y_true)
                loss_val += loss.data  # [0]
                loss.backward()
                self.W1.data -= self.learning_rate * self.W1.grad.data
                self.W2.data -= self.learning_rate * self.W2.grad.data

                self.W1.grad.data.zero_()
                self.W2.grad.data.zero_()
            if epo % 10 == 0:
                print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')


class CBOW(nn.Module):
    def __init__(self, vocab_size, embd_size, context_size, hidden_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)
        self.linear1 = nn.Linear(2 * context_size * embd_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embedded = self.embeddings(inputs).view((1, -1))
        hid = F.relu(self.linear1(embedded))
        out = self.linear2(hid)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embd_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)

    def forward(self, focus, context):
        embed_focus = self.embeddings(focus).view((1, -1))
        embed_ctx = self.embeddings(context).view((1, -1))
        score = torch.mm(embed_focus, torch.t(embed_ctx))
        log_probs = F.logsigmoid(score)

        return log_probs


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
