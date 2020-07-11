import os, sys
import pandas as pd

basedir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(basedir)

class ManageData(object):

    def __init__(self, src_char2id, max_len=None, filelist=None):
        self.max_len = max_len
        self.filelist = filelist

        self.char_dict = self._make_char_dict(src_char2id)
        self.unk_id = self.char_dict.get('[UNK]')


    def _make_char_dict(self, src_char2id):
        """ make char2id dictionary based on input file
            src_char2id: \t splitted char2id file
            return: {'char':id, ... } dictionary
        """
        with open(src_char2id,'r') as f:
            lines = f.read().splitlines()
        return {line.split('\t')[1]:int(line.split('\t')[0]) for line in lines}


    def preprocess(self, sentence, max_len=None, label=0):
        """ convert sentence to character_id list"""
        if max_len is None: max_len = self.max_len

        # char 2 id
        char_id_list = [self.char_dict.get(c,self.unk_id) for c in sentence]

        # trim to max len
        char_id_list = char_id_list[:max_len]
        sequence_length = len(char_id_list)

        # pad to max len
        char_id_list.extend([0]*(max_len-sequence_length))

        return {"sentence":sentence,
                "X":char_id_list,
                "Y":int(label),
                "sequence_length":sequence_length}


    def generator(self):
        """ return generator yields preprocessed data
            filelist: tsv based data with 'document', 'label' col
        """
        for filename in self.filelist:
            # parse tsv
            dataset = pd.read_csv(filename, delimiter='\t', header=0)
            for i in range(len(dataset)):
                try:
                    sentence = dataset['document'][i]
                    label = dataset['label'][i]

                    yield self.preprocess(sentence, self.max_len, label)

                except:
                    continue


if __name__ == "__main__":
    md = ManageData('data/char2id.txt')

    sentence = 'test문장 입니다.'
    print(md.preprocess(sentence,1,150))

