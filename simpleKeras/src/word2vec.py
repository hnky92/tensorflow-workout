from gensim.models import Word2Vec
from gensim.models import KeyedVectors

class W2VTrainer():
    def __init__(self, src_train_data):

        from src.generator import ManageData
        md = ManageData()

        sentences, _ = md.parse_data(src_train_data)

        self.train_data = []
        for sentence in sentences:
            res = md.preprocess(sentence)
            if res is not None:
                self.train_data.append(res)


    def train(self, epochs=15, min_count=50, size=256, savedir='data/w2v.model'):

        model = Word2Vec(self.train_data,
                         min_count=min_count, 
                         size=size, 
                         window=3, 
                         workers=4,
                         iter=epochs)

        model.save(savedir)


if __name__ == "__main__":
    model = KeyedVectors.load('data/w2v.model')

    print(model.most_similar('감독'))
    print(model.most_similar('평점'))
    print(model.most_similar('액션'))
    print(model.most_similar('배우'))
    print(model.most_similar('재미'))

    for item in model.wv.vocab:
        print( model.wv.vocab[item].count)


    print(len(model.wv.vocab))
