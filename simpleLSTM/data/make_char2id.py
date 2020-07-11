import pandas as pd
from tqdm import tqdm

from collections import Counter

def process(src_in, src_out, limit=2000):
    """ record most n common characters """

    max_len = 0
    dataset = pd.read_csv("ratings_train.txt", delimiter='\t', header=0)

    counter = Counter()
    for i in tqdm(range(len(dataset))):
        try:
            max_len = max(max_len,len(dataset['document'][i]))
            counter.update(dataset['document'][i])
        except:
            continue
        
    print("##maxlen : %s" % (max_len))

    most_common = counter.most_common()
    most_common = most_common[:limit-1]

    with open(src_out,'w') as f_out: 
        f_out.write("%s\t%s\t%s\n" % (0,'[UNK]',0))

        for i,tup in tqdm(enumerate(most_common)): 
            f_out.write("%s\t%s\t%s\n" % (i+1,tup[0],tup[1]))


if __name__ == '__main__':
    src_in = 'ratings_train.txt'
    src_out = 'char2id.txt'
    process(src_in, src_out)
