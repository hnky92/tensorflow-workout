from tensorflow import keras
from tensorflow.keras import layers
from gensim.models import KeyedVectors

from absl import flags

from src.utils import load_token_id


import numpy as np

FLAGS = flags.FLAGS

class LSTMModel(keras.Model):
    """
    1. load w2v embedding -> init embedding layer
    2. bidirectional lstm output
    """

    def __init__(self):
        super(LSTMModel, self).__init__()

        embeddings = self._load_w2v_matrix()

        self.embed_layer = layers.Embedding(FLAGS.vocab_size+2, 
                                            FLAGS.vec_dim, 
                                            weights=[embeddings],
                                            input_length=FLAGS.max_len,
                                            mask_zero=True)

        self.bidi_lstm = layers.Bidirectional(layers.LSTM(FLAGS.vec_dim, 
                                                          return_sequences=False))

        self.dense_out = layers.Dense(FLAGS.num_class)


    def _load_w2v_matrix(self):
        token_dict = load_token_id(FLAGS.src_token2id)
        token_dict_rev = {v:k for k,v in token_dict.items()}

        word2vec_model = KeyedVectors.load(FLAGS.src_w2v)

        default_vec = np.asarray([0]*FLAGS.vec_dim)
        embeddings = np.vstack((default_vec,default_vec))
        for i in range(2,len(token_dict_rev.keys())+1):
            embeddings = np.vstack((embeddings,word2vec_model.wv[token_dict_rev[i]]))

        return embeddings



    def call(self, inputs):
        x = self.embed_layer(inputs)
        x = self.bidi_lstm(x)
        x = self.dense_out(x)

        return x

