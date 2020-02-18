import numpy
from keras.preprocessing.sequence import pad_sequences

from datautils import DataSequence
from preprocessing import pad_nested_sequences

class Trainer():
    """ Trains a model and evaluates its performance on a test dataset
    """

    def __init__(self, model, X, y, preprocessor, split=[0.8, 0.95]):
        self.model = model
        self.x_train, self.x_val, self.x_test = numpy.split(X, [int(len(X)*split[0]), int(len(X)*split[1])])
        self.y_train, self.y_val, self.y_test = numpy.split(y, [int(len(X)*split[0]), int(len(X)*split[1])])
        self.preprocessor = preprocessor

    def train(self, batch_size, epochs):
        """ Trains the model using the training and validation datasets
        """
        train_seq = DataSequence(self.x_train, self.y_train, batch_size, self.preprocessor)
        val_seq = DataSequence(self.x_val, self.y_val, batch_size, self.preprocessor)

        self.model.train(train_seq, val_seq, epochs)

    def evaluate(self, idx2label):
        """ Evaluates the model on the test dataset
        """
        wrong = 0
        for sentence, true_labels in zip(self.x_test, self.y_test):
            features = self.preprocessor([sentence])
            p = self.model.predict(features)

            predicted_labels = []
            for pred in p:
                predicted_labels.append(idx2label[pred])

            if predicted_labels != true_labels:
                wrong += 1

        percentage = 100*((len(self.x_test)-wrong) / len(self.x_test))
        print("Testset accuracy is %s percent" % percentage)