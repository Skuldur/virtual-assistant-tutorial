# -*- coding: utf-8 -*-
import json
from os import path
import io
import argparse
import numpy
import json
from keras.preprocessing.sequence import pad_sequences
from layers import CRF
from sequences import TrainSequence
from preprocessing import IndexTransformer, pad_nested_sequences
from model import BiLSTMCRF

def get_word_id(vocab, word):
    if word in vocab:
        return vocab[word]
    return vocab['<unk>']

def get_char_id(vocab, char):
    if char in vocab:
        return vocab[char]
    return vocab['<unk>']

class IntentTrainer():

    def __init__(self, schema_path, name, lang='en'):
        """ Gets the schema for the new intent passed in and parses it """
        self.schema_path = schema_path
        self.name = name
        self.vocab = set()
        self.vocab.add("<unk>")

    def save_data(self, example_commands, labels, vocab_map, char_map):
        data_path = path.join(path.dirname(path.realpath(__file__)), 'data', '%s.json' % self.name)
        with open(data_path, 'w', encoding='utf-8') as fp:
            json.dump(example_commands, fp, ensure_ascii=False)

        label_path = path.join(path.dirname(path.realpath(__file__)), "intents",  "config", "labels", "%s_labels.json" % self.name)
        with open(label_path, 'w', encoding='utf-8') as fp:
            json.dump(labels, fp, ensure_ascii=False)

        word_path = path.join(path.dirname(path.realpath(__file__)), "intents", "config", "vocab", "%s_word_vocab.json" % self.name)
        with open(word_path, 'w', encoding='utf-8') as fp:
            json.dump(vocab_map, fp, ensure_ascii=False)

        char_path = path.join(path.dirname(path.realpath(__file__)), "intents", "config", "vocab", "%s_char_vocab.json" % self.name)
        with open(char_path, 'w', encoding='utf-8') as fp:
            json.dump(char_map, fp, ensure_ascii=False)

    def add_to_vocab(self, word):
        self.vocab.add(word)

    def get_data(self):
        with open(self.schema_path) as f:
            data = json.load(f)

        return data['labels'], data['training_data']

    def build_vocabulary(self, X):
        for command in X:
            for word in command:
                self.add_to_vocab(word)

    def train(self):

        labels, data = self.get_data()

        X = [x['words'] for x in data]
        Y = [x['labels'] for x in data]

        self.build_vocabulary(X)

        n_words = len(self.vocab)

        vocab = sorted(list(self.vocab))

        vocab_map = dict((word, number) for number, word in enumerate(vocab))

        #char embedding
        chars = sorted(list(set([w_i for w in vocab for w_i in w])))
        n_chars = len(chars)

        char2idx = {c: i + 2 for i, c in enumerate(chars)}
        char2idx["<unk>"] = 1
        char2idx["<pad>"] = 0

        labels2idx = dict((label, number) for number, label in enumerate(labels))

        self.save_data(X[:300], labels, vocab_map, char2idx)

        x_train, x_val, x_test = numpy.split(X, [int(len(X)*0.75), int(len(X)*0.95)])
        y_train, y_val, y_test = numpy.split(Y, [int(len(X)*0.75), int(len(X)*0.95)])
        batch_size = 64

        preprocessor = IndexTransformer()
        preprocessor.fit(vocab_map, labels2idx, char2idx)

        train_seq = TrainSequence(x_train, y_train, batch_size, preprocessor.transform)
        val_seq = TrainSequence(x_val, y_val, batch_size, preprocessor.transform)

        model = BiLSTMCRF(labels, n_words, n_chars)
        model.build()
        model.compile()
        model.train(train_seq, val_seq)

        weights_path = path.join(path.dirname(path.realpath(__file__)), "intents", "config", "weights", '%s.hdf5' % self.name)
        model.model.save_weights(weights_path)

        vocab_map = dict((word, number) for number, word in enumerate(vocab))

        idx2label = dict((number, label) for number, label in enumerate(labels))

        sentences = x_test

        wrong = 0

        for sentence, true_labels in zip(sentences, y_test):
            words = [w for w in sentence]
            word_id_array = [[get_word_id(vocab_map, w) for w in sentence]]
            word_id_array = pad_sequences(sequences=word_id_array, padding="post", value=n_words)
            
            char_ids = [[[get_char_id(char2idx, ch) for ch in w] for w in sentence]]
            char_ids = pad_nested_sequences(char_ids)
            p = model.predict([numpy.array(word_id_array), numpy.array(char_ids)])

            predicted_labels = []
            for pred in p[0]:
                predicted_labels.append(idx2label[pred])

            if predicted_labels != true_labels:
                wrong += 1

        percentage = 100*(1.0*(len(sentences)-wrong) / len(sentences))
        print("Testset accuracy is %s percent" % percentage)


parser = argparse.ArgumentParser(description='Train a new intent')
parser.add_argument('schema_file', type=str,
                    help='The path to the intent schema')
parser.add_argument('name', type=str,
                    help='The path to the intent schema')


def run(schema_path, name):
    trainer = IntentTrainer(schema_path, name)
    trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.schema_file, args.name)

    