from os import path
import math
import json

import numpy as np
from keras.utils import Sequence


class Dataset():
    def __init__(self, schema_path, name):
        self.path = schema_path
        self.name = name

        with open(self.path) as f:
            data = json.load(f)

        self.labels = data['labels']
        self.training_data = data['training_data'][:10000]

    def get_data(self):
        return self.labels, self.training_data 

    def save(self, example_commands, labels):
        data_path = path.join(path.dirname(path.realpath(__file__)), 'data', '%s.json' % self.name)
        with open(data_path, 'w', encoding='utf-8') as fp:
            json.dump(example_commands, fp, ensure_ascii=False)

        label_path = path.join(path.dirname(path.realpath(__file__)), "intents",  "config", "labels", "%s_labels.json" % self.name)
        with open(label_path, 'w', encoding='utf-8') as fp:
            json.dump(labels, fp, ensure_ascii=False)


class Vocabulary():
    def __init__(self):
        self.vocab = set()
        self.vocab.add('<pad>')
        self.vocab.add('<unk>')
        self.map = None
        self._create_map()

    def add(self, word):
        self.vocab.add(word)
        self._create_map()

    def build_vocab(self, X):
        for command in X:
            for word in command:
                self.vocab.add(word)

        self._create_map()

    def _create_map(self):
        l = sorted(list(self.vocab))
        self.map = {word: number for number, word in enumerate(l)}

    def get_map(self):
        return self.map

    def get(self, word):
        return self.map.get(word, self.map['<unk>'])

    def save(self, name):
        vocab_path = path.join(path.dirname(path.realpath(__file__)), "intents", "config", "vocab", name)
        with open(vocab_path, 'w', encoding='utf-8') as fp:
            json.dump(self.map, fp, ensure_ascii=False)


    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, word):
        return self.get(word)

    def __contains__(self, word):
        return word in self.vocab


class DataSequence(Sequence):
    def __init__(self, x, y, batch_size=1, preprocess=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.preprocess = preprocess

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return self.preprocess(batch_x, batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)