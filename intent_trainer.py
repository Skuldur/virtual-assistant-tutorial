# -*- coding: utf-8 -*-
import argparse
from os import path
import json

from nltk.tokenize import word_tokenize as tokenize

from trainer import Trainer
from datautils import Dataset, Vocabulary
from preprocessing import Preprocessor
from model import BiLSTMCRF


parser = argparse.ArgumentParser(description='Train a new intent')
parser.add_argument('task', nargs='?', type=str, default='train',
                    help='The task we want to run')
parser.add_argument('-n', '--name', type=str,
                    help='The path to the intent schema')

#Train arguments
parser.add_argument('-f', '--schema_file', type=str,
                    help='The path to the intent schema')
parser.add_argument('-s', '--sample_size', type=int,
                    help='The path to the intent schema', default=300)
parser.add_argument('-b', '--batch_size', type=int,
                    help='The path to the intent schema', default=64)
parser.add_argument('-e', '--epochs', type=int,
                    help='The path to the intent schema', default=10)

#Predict arguments
parser.add_argument('--command', type=str, 
                    help='A command for our intent', default='')


def run(schema_path, name, sample_size, batch_size, epochs):
    dataset = Dataset(schema_path, name)
    labels, data = dataset.get_data()

    X = [x['words'] for x in data]
    y = [x['labels'] for x in data]

    word_vocab = Vocabulary()
    word_vocab.build_vocab([w for command in X for w in command])

    #char embedding
    char_vocab = Vocabulary()
    char_vocab.build_vocab([ch for w in word_vocab for ch in w])

    labels2idx = dict((label, idx) for idx, label in enumerate(labels))
    idx2label = dict((idx, label) for idx, label in enumerate(labels))

    preprocessor = Preprocessor(word_vocab, labels2idx, char_vocab)
    model = BiLSTMCRF(labels, len(word_vocab), len(char_vocab))
    trainer = Trainer(model, X, y, preprocessor.transform, split=[0.75, 0.95])

    trainer.train(batch_size, epochs)
    trainer.evaluate(idx2label)

    model.save_weights(name)
    dataset.save(X[:sample_size], labels)
    word_vocab.save("%s_word_vocab.json" % name)
    char_vocab.save("%s_char_vocab.json" % name)

def predict(name, command):
    command = command.lower()

    label_path = path.join(
        path.dirname(path.realpath(__file__)), 
        "intents", 
        "config", 
        "labels", 
        "%s_labels.json" % name
    )
    with open(label_path, encoding="utf8") as f:
        labels = json.load(f)

    word_vocab = Vocabulary()
    word_vocab.load("%s_word_vocab.json" % name)

    #char embedding
    char_vocab = Vocabulary()
    char_vocab.load("%s_char_vocab.json" % name)

    idx2label = dict((idx, label) for idx, label in enumerate(labels))

    preprocessor = Preprocessor(word_vocab, None, char_vocab)
    model = BiLSTMCRF(labels, len(word_vocab), len(char_vocab))
    model.load_weights('intents/config/weights/%s.hdf5' % name)

    sentence = tokenize(command)
    features = preprocessor.transform([sentence])

    p = model.predict(features)
    predicted_labels = []
    for pred in p:
        predicted_labels.append(idx2label[pred])

    for word, label in zip(sentence, predicted_labels):
        print('%s: %s' % (word, label))


if __name__ == '__main__':
    args = parser.parse_args()

    if args.task == 'train':
        run(
            args.schema_file, 
            args.name, 
            args.sample_size,
            args.batch_size,
            args.epochs
        )
    elif args.task == 'predict':
        predict(
            args.name,
            args.command
        )
    else:
        raise RuntimeError('Task not supported')

    