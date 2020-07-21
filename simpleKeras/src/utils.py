import json

def padding(sequence, max_len=128, pad_token=0):
    if max_len < len(sequence):
        return sequence[:max_len]

    len_to_pad = max_len - len(sequence)

    return sequence + [0]*len_to_pad


def tokenlist2idlist(tokenlist, token_dict):
    def map_token(token):
        nonlocal token_dict

        try:
            return token_dict[token]
        except:
            return token_dict['UNK']

    return [map_token(token) for token in tokenlist]
    

def load_token_id(src_token2id):
    """
    load token2id json file 
    ex) {'3873': '소피마르소', '3874': '밴드', ... } 
    return as dictionary
    """
    with open(src_token2id, 'r') as f:
        return json.load(f)


def make_token_id(src_token2id, src_w2v, vocab_size):
    """
    form trained w2v. make token2id mapping file
    """
    from gensim.models import KeyedVectors

    model = KeyedVectors.load('data/w2v.model')

    # load w2v
    tmp = []
    for item in model.wv.vocab:
        tmp.append((item, model.wv.vocab[item].count))

    tmp.sort(key=lambda el:el[1], reverse=True)

    # to dict
    token_dict = dict()
    for i, element in enumerate(tmp[:vocab_size]):
        token_dict[element[0]] = i+2

    # UNK: 1
    # reserve 0 for PAD
    token_dict['UNK'] = 1

    with open(src_token2id, 'w')  as f:
        json.dump(token_dict, f, ensure_ascii=False, indent=4)


