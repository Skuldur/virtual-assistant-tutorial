# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2017 Hiroki Nakayama

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Preprocessors.
"""
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

class IndexTransformer(BaseEstimator, TransformerMixin):
    """Convert a collection of raw documents to a document id matrix.  
    """
    
    def fit(self, vectors, labels2idx, chars2idx):
        """Learn vocabulary from training set.

        Args:
            X : iterable. An iterable which yields either str, unicode or file objects.

        Returns:
            self : IndexTransformer.
        """
        self._char_vocab = chars2idx
        self._word_vocab = vectors
        self._label_vocab = labels2idx

        return self

    def transform(self, X, y=None):
        """Transform documents to document ids.

        Uses the vocabulary learned by fit.

        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.
            y : iterabl, label strings.

        Returns:
            features: document id matrix.
            y: label id matrix.
        """

        # Convert every word in each sentence into an integer id
        word_ids = [[self._word_vocab[w] for w in doc] for doc in X]
        n_words = len(self._word_vocab)

        # Pad all the sequences to equal length.
        # The padding value is #n_words so that the padding value does not match
        # any of the proper word ids
        word_ids = pad_sequences(sequences=word_ids, padding="post", value=self._word_vocab['<pad>'])

        # Convert every character of every word in each sentence into an integer id
        char_ids = [[[self._char_vocab[ch] for ch in w] for w in doc] for doc in X]
        # Each word gets its own array so we need to pad those arrays so that 
        # each word sequence is of the same length
        char_ids = pad_nested_sequences(char_ids)

        features = [word_ids, char_ids]

        if y is not None:
            # Convert every label into an integer id
            y = [[self._label_vocab[label] for label in doc] for doc in y]
            # Pad all the sequences so they match the padded word arrays
            # The padding value is 0 because that will always be 'O' label 
            y = pad_sequences(y, padding='post', value=0)
            # One-hot encode all the sequences
            y = to_categorical(y, self.label_size).astype(int)
            return features, y

    def fit_transform(self, X, y=None, **params):
        """Learn vocabulary and return document id matrix.

        This is equivalent to fit followed by transform.

        Args:
            X : iterable
            an iterable which yields either str, unicode or file objects.

        Returns:
            list : document id matrix.
            list: label id matrix.
        """
        return self.fit(X, y).transform(X, y)

    @property
    def word_vocab_size(self):
        return len(self._word_vocab)

    @property
    def char_vocab_size(self):
        return len(self._char_vocab)

    @property
    def label_size(self):
        return len(self._label_vocab)

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)

        return p

def pad_nested_sequences(sequences, dtype='float32'):
    """Pads nested sequences to the same length.

    This function transforms a list of list sequences
    into a 3D Numpy array of shape `(num_samples, max_sent_len, max_word_len)`.

    Args:
        sequences: List of lists of lists.
        dtype: Type of the output sequences.

    # Returns
        x: Numpy array.
    """
    max_sent_len = 0
    max_word_len = 0
    for sent in sequences:
        max_sent_len = max(len(sent), max_sent_len)
        for word in sent:
            max_word_len = max(len(word), max_word_len)

    x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)
    for i, sent in enumerate(sequences):
        for j, word in enumerate(sent):
            x[i, j, :len(word)] = word

    return x