# -*- coding: utf-8 -*-
from os import path
import argparse

import numpy
from keras.preprocessing.sequence import pad_sequences

from datautils import Dataset, Vocabulary, DataSequence
from preprocessing import IndexTransformer, pad_nested_sequences
from model import BiLSTMCRF


class Trainer():
    def __init__(self, model, X, y, split=[0.8, 0.95]):
        self.model = model
        self.x_train, self.x_val, self.x_test = numpy.split(X, [int(len(X)*split[0]), int(len(X)*split[1])])
        self.y_train, self.y_val, self.y_test = numpy.split(y, [int(len(X)*split[0]), int(len(X)*split[1])])

    def train(self, batch_size, preprocessor):
        train_seq = DataSequence(self.x_train, self.y_train, batch_size, preprocessor.transform)
        val_seq = DataSequence(self.x_val, self.y_val, batch_size, preprocessor.transform)

        self.model.build()
        self.model.compile()
        self.model.train(train_seq, val_seq)

    def evaluate(self, word2idx, char2idx, idx2label):
        sentences = self.x_test

        wrong = 0
        for sentence, true_labels in zip(sentences, self.y_test):
            words = [w for w in sentence]
            word_id_array = [[word2idx[w] for w in sentence]]
            word_id_array = pad_sequences(sequences=word_id_array, padding="post", value=word2idx['<pad>'])
            
            char_ids = [[[char2idx[ch] for ch in w] for w in sentence]]
            char_ids = pad_nested_sequences(char_ids)
            p = self.model.predict([numpy.array(word_id_array), numpy.array(char_ids)])

            predicted_labels = []
            for pred in p[0]:
                predicted_labels.append(idx2label[pred])

            if predicted_labels != true_labels:
                wrong += 1

        percentage = 100*((len(sentences)-wrong) / len(sentences))
        print("Testset accuracy is %s percent" % percentage)


class IntentTrainer():

    def __init__(self, schema_path, name, lang='en'):
        """ Gets the schema for the new intent passed in and parses it """
        self.schema_path = schema_path
        self.name = name
        

    def train(self):
        dataset = Dataset(self.schema_path, self.name)
        labels, data = dataset.get_data()

        X = [x['words'] for x in data]
        y = [x['labels'] for x in data]

        word_vocab = Vocabulary()
        word_vocab.build_vocab(X)

        #char embedding
        char_vocab = Vocabulary()
        char_vocab.build_vocab([ch for w in word_vocab.vocab for ch in w])

        labels2idx = dict((label, idx) for idx, label in enumerate(labels))
        idx2label = dict((idx, label) for idx, label in enumerate(labels))

        dataset.save(X[:300], labels, word_vocab, char_vocab)
        model = BiLSTMCRF(labels, len(word_vocab), len(char_vocab))

        trainer = Trainer(model, X, y, [0.75, 0.95])

        batch_size = 64
        preprocessor = IndexTransformer()
        preprocessor.fit(word_vocab, labels2idx, char_vocab)

        trainer.train(batch_size, preprocessor)
        model.save_weights(self.name)
        trainer.evaluate(word_vocab, char_vocab, idx2label)


parser = argparse.ArgumentParser(description='Train a new intent')
parser.add_argument('schema_file', type=str,
                    help='The path to the intent schema')
parser.add_argument('name', type=str,
                    help='The path to the intent schema')


def run(schema_path, name):
    trainer = IntentTrainer(schema_path, name)
    trainer.train()


if __name__ == '__main__':
    #args = parser.parse_args()
    #run(args.schema_file, args.name)
    run('commands/play_commands.json', 'play')

    