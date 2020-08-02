import os, sys
import pandas as pd

import tensorflow as tf

from konlpy.tag import Mecab

from src.utils import load_token_id
from src.utils import tokenlist2idlist
from src.utils import padding

#basedir = os.path.split(os.path.abspath(__file__))[0]
#sys.path.append(basedir)

class ManageData(object):

    def __init__(self, src_token_dict=None, filelist=None, max_len=128, num_class=2):
        self.filelist = filelist
        self.max_len = max_len
        self.num_class = num_class

        self.mecab = Mecab()

        if src_token_dict is not None:
            self.token_dict = load_token_id(src_token_dict)


    def preprocess(self, sentence):
        try:
            return self.mecab.morphs(sentence)
        except:
            return None


    def parse_data(self, filename):
        dataset = pd.read_csv(filename, delimiter='\t', header=0)
        sentence_list = dataset['document']
        label_list = dataset['label']

        return sentence_list, label_list

    def _to_onehot(self, label):
        tmp = [0]*self.num_class
        tmp[label] = 1
        return tmp


    def generator(self):
        """ return generator yields preprocessed data
            filelist: tsv based data with 'document', 'label' col
        """
        for filename in self.filelist:
            # parse tsv
            sentences, labels = self.parse_data(filename)

            for sentence, label in zip(sentences, labels):
                try:
                    processed = self.preprocess(sentence)
                    processed = tokenlist2idlist(processed, self.token_dict)
                    processed = padding(processed, self.max_len) 

                    if processed == None: continue

                    yield (processed, self._to_onehot(label))
                except:
                    continue


if __name__ == "__main__":
    md = ManageData('data/token2id.json',['data/ratings_train.txt'])

    gen = md.generator()
    for i in range(100):
        print(next(gen))


