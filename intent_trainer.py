# -*- coding: utf-8 -*-
import argparse

from trainer import Trainer
from datautils import Dataset, Vocabulary
from preprocessing import Preprocessor
from model import BiLSTMCRF


parser = argparse.ArgumentParser(description='Train a new intent')
parser.add_argument('--schema_file', type=str,
                    help='The path to the intent schema')
parser.add_argument('--name', type=str,
                    help='The path to the intent schema')
parser.add_argument('--sample_size', type=int,
                    help='The path to the intent schema', default=300)
parser.add_argument('--batch_size', type=int,
                    help='The path to the intent schema', default=64)

def run(schema_path, name, sample_size, batch_size):
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
    trainer = Trainer(model, X, y, [0.75, 0.95])

    trainer.train(batch_size, preprocessor.transform)
    trainer.evaluate(word_vocab, char_vocab, idx2label)

    model.save_weights(name)
    dataset.save(X[:sample_size], labels)
    word_vocab.save("%s_word_vocab.json" % name)
    char_vocab.save("%s_char_vocab.json" % name)

if __name__ == '__main__':
    args = parser.parse_args()
    run(args.schema_file, 
        args.name, 
        args.sample_size,
        args.batch_size
    )

    