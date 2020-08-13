from tensorflow import keras
from tensorflow.keras import layers

from gensim.models import KeyedVectors
from src.utils import load_token_id

import numpy as np


class LSTMModel(keras.Model):
    """
    1. load w2v embedding -> init embedding layer
    2. bidirectional lstm
    3. dense layer
    """

    def __init__(self, vocab_size=4000, vec_dim=256, max_len=128, num_class=2, train=True, src_token2id=None, src_w2v=None):
        super(LSTMModel, self).__init__()

        self.max_len = max_len
        self.vec_dim = vec_dim

        if train:
            embeddings = self._load_w2v_matrix(src_token2id, src_w2v, vec_dim)
        else:
            embeddings = np.zeros((vocab_size+2, vec_dim))

        self.embed_layer = layers.Embedding(vocab_size+2, 
                                            vec_dim, 
                                            weights=[embeddings],
                                            input_length=max_len,
                                            mask_zero=True)

        self.bidi_lstm = layers.Bidirectional(layers.LSTM(vec_dim, 
                                                          return_sequences=False))

        self.dense_out = layers.Dense(num_class)


    def _load_w2v_matrix(self, src_token2id, src_w2v, vec_dim):
        """
        load pretrained word2vec matrix
        """
        token_dict = load_token_id(src_token2id)
        token_dict_rev = {v:k for k,v in token_dict.items()}

        word2vec_model = KeyedVectors.load(src_w2v)

        default_vec = np.asarray([0]*vec_dim)
        embeddings = np.vstack((default_vec,default_vec))
        for i in range(2,len(token_dict_rev.keys())+1):
            embeddings = np.vstack((embeddings,word2vec_model.wv[token_dict_rev[i]]))

        return embeddings


    def call(self, inputs):
        """
        forward pass
        """
        x = self.embed_layer(inputs)
        x = self.bidi_lstm(x)
        x = self.dense_out(x)

        return x


    def model(self):
        """
        load model with input, softmaxoutput layer
        """
        x = keras.Input(shape=(self.max_len,))
        logit = self.call(x)
        pred = keras.layers.Softmax()(logit)

        return keras.Model(inputs=[x], outputs=pred)


