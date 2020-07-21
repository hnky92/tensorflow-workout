from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('job', None, 'job to do')
flags.mark_flag_as_required('job')

## directories
flags.DEFINE_string('src_train_data','data/ratings_train.txt','train data')
flags.DEFINE_string('src_valid_data','data/ratings_test.txt','valid data')
flags.DEFINE_string('src_token2id','data/token2id.json','token 2 id map')
flags.DEFINE_string('src_w2v','data/w2v.model','word2vec model')

flags.DEFINE_string('checkpoint_dir','checkpoints','dir to save model')

## model params
flags.DEFINE_integer('epochs', 10, 'max epochs to train model')
flags.DEFINE_integer('vocab_size', 4000, 'max vocab size to use')
flags.DEFINE_integer('vec_dim', 256, 'word vector dimension')
flags.DEFINE_integer('max_len', 128, 'max token sequence length of sentence')
flags.DEFINE_integer('num_class', 2, 'number of classification class')
flags.DEFINE_integer('num_train_data', 199001, 'number of train data')
flags.DEFINE_integer('batch_size', 32, 'train batch size')


def main(argv):
    """ select process to do """
    if FLAGS.job == 'w2v':
        from src.word2vec import W2VTrainer
        W2VTrainer(FLAGS.src_train_data).train(savedir=FLAGS.src_w2v)

    if FLAGS.job == 't2id':
        from src.utils import make_token_id
        make_token_id(FLAGS.src_token2id, FLAGS.src_w2v, FLAGS.vocab_size)


    if FLAGS.job == 'train':
        from src.trainer import Trainer
        Trainer().train()


if __name__ == "__main__":
    app.run(main)
    

