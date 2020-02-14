import numpy

from keras.layers import Dense, CuDNNLSTM, LSTM, concatenate, SpatialDropout1D, Bidirectional, Embedding, Input, Dropout, TimeDistributed, GlobalAveragePooling1D
from keras.layers.merge import Concatenate
from keras.models import Model
from layers import CRF
from keras.callbacks import ModelCheckpoint


class BiLSTMCRF():

    def __init__(self, labels, n_words, n_chars=0, dropout=0.3):
        self.n_labels = len(labels)
        self.n_words = n_words
        self.dropout = dropout
        self.n_chars = n_chars

    def build(self):
        # build word embedding
        word_in = Input(shape=(None,))
        emb_word = Embedding(input_dim=self.n_words+1, output_dim=100)(word_in)
        
        char_in = Input(shape=(None, None,))
        emb_char = TimeDistributed(Embedding(input_dim=self.n_chars + 2, output_dim=10,
                         mask_zero=True))(char_in)
        # character LSTM to get word encodings by characters
        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                        recurrent_dropout=0.5))(emb_char)


        x = concatenate([emb_word, char_enc])
        x = SpatialDropout1D(0.3)(x)
        bi_lstm = Bidirectional(LSTM(units=256, return_sequences=True,
                               recurrent_dropout=0.6))(x)


        fully_conn = TimeDistributed(Dense(self.n_labels, activation="relu"))(bi_lstm)  # softmax output layer

        crf = CRF(self.n_labels, sparse_target=False)
        loss = crf.loss_function
        pred = crf(fully_conn)

        self.model = Model(inputs=[word_in, char_in], outputs=pred)
        self.loss = loss
        self.accuracy = crf.accuracy

    def compile(self):
        self.model.compile(loss=self.loss, optimizer='rmsprop', metrics=[self.accuracy])

    def train(self, train_seq, test_seq):
        self.model.fit_generator(
            generator=train_seq,
            epochs=10,
            verbose=1,
            shuffle=True,
            validation_data=test_seq,
        )

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def save_weights(self, name):
        self.model.save_weights(name)

    def predict(self, input):
        p = self.model.predict(input)
        p = numpy.argmax(p, axis=-1)

        return p


class TextClassification():

    def __init__(self, labels, n_words, dropout=0.3):
        self.n_labels = len(labels)
        self.n_words = n_words
        self.dropout = dropout

    def build(self):
        # build word embedding
        input = Input(shape=(None,))
        model = Embedding(input_dim=self.n_words+1, output_dim=50)(input)
        #model = Dropout(self.dropout)(model)
        model = GlobalAveragePooling1D()(model)
        out = Dense(256, activation="relu")(model)  # softmax output layer
        out = Dense(self.n_labels, activation="softmax")(model)  # softmax output layer

        self.model = Model(inputs=input, outputs=out)

        return model

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def train(self, train_seq, test_seq):
        self.model.fit_generator(
            generator=train_seq,
            epochs=30,
            verbose=1,
            shuffle=True,
            validation_data=test_seq,
        )

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def predict(self, input):
        p = self.model.predict(numpy.array(input))
        print(p)
        p = numpy.argmax(p, axis=-1)

        return p