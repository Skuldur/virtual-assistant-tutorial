import numpy
from keras.preprocessing.sequence import pad_sequences

from datautils import DataSequence
from preprocessing import pad_nested_sequences

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
        wrong = 0
        for sentence, true_labels in zip(self.x_test, self.y_test):
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

        percentage = 100*((len(self.x_test)-wrong) / len(self.x_test))
        print("Testset accuracy is %s percent" % percentage)