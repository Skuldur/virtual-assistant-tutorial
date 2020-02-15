# -*- coding: utf-8 -*-
import argparse

from trainer import Trainer
from datautils import Dataset, Vocabulary
from preprocessing import Preprocessor
from model import BiLSTMCRF


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

        model = BiLSTMCRF(labels, len(word_vocab), len(char_vocab))
        trainer = Trainer(model, X, y, [0.75, 0.95])

        preprocessor = Preprocessor(word_vocab, labels2idx, char_vocab)

        batch_size = 64
        trainer.train(batch_size, preprocessor)
        trainer.evaluate(word_vocab, char_vocab, idx2label)

        model.save_weights(self.name)
        dataset.save(X[:300], labels)
        word_vocab.save("%s_word_vocab.json" % self.name)
        char_vocab.save("%s_char_vocab.json" % self.name)


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

    