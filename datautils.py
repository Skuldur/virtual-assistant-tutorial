from os import path
import math
import json

import numpy as np
from keras.utils import Sequence


CONFIG_PATH = path.join(path.dirname(path.realpath(__file__)), "intents", "config")


class Dataset():
    def __init__(self, schema_path, name):
        self.path = schema_path
        self.name = name

        with open(self.path) as f:
            data = json.load(f)

        self.labels = data['labels']
        self.training_data = data['training_data']

    def get_data(self):
        return self.labels, self.training_data 

    def save(self, example_commands, labels):
        data_path = path.join(path.dirname(path.realpath(__file__)), 'data', '%s.json' % self.name)
        with open(data_path, 'w', encoding='utf-8') as fp:
            json.dump(example_commands, fp, ensure_ascii=False)

        label_path = path.join(CONFIG_PATH, "labels", "%s_labels.json" % self.name)
        with open(label_path, 'w', encoding='utf-8') as fp:
            json.dump(labels, fp, ensure_ascii=False)


class Vocabulary():
    def __init__(self):
        self.vocab = {}
        self.length = 0
        self.add('<pad>')
        self.add('<unk>')

    def add(self, word):
        if not word in self.vocab:
            self.vocab[word] = self.length
            self.length += 1

    def build_vocab(self, items):
        for item in items:
            self.add(item)

    def get(self, item):
        """Gets the integer ID of the item if it exists in the vocabulary,
        Otherwise it gets the ID of the unknown token
        """
        return self.vocab.get(item, self.vocab['<unk>'])

    def save(self, name):
        vocab_path = path.join(CONFIG_PATH, "vocab", name)
        with open(vocab_path, 'w', encoding='utf-8') as fp:
            json.dump(self.vocab, fp, ensure_ascii=False)

    def load(self, name):
        vocab_path = path.join(CONFIG_PATH, "vocab", name)
        with open(vocab_path, encoding="utf8") as f:
            self.vocab = json.load(f)
            self.length = len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item):
        return self.get(item)

    def __contains__(self, item):
        return item in self.vocab

    def __iter__(self):
        for w in self.vocab:
            yield w


class DataSequence(Sequence):
    def __init__(self, x, y, batch_size=1, preprocess=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.preprocess = preprocess

    def __getitem__(self, idx):
        """Generates the next batch for the model and returns the preprocessed data
        """
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return self.preprocess(batch_x, batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)