import tensorflow as tf

from src.model import LSTMModel
from src.generator import ManageData

class Wrapper(object):

    def __init__(self, src_model, src_token2id):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                print(e)

        self.MD = ManageData(src_token2id)

        self.model = LSTMModel(train=False).model()

        self.model.load_weights(src_model)


    def predict(self, sentence):
        """
        predict model result
        """
        vec = self.MD.make_input(sentence)

        return self.model.predict([vec])[0]



