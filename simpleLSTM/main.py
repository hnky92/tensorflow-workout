from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('job', None, 'job to do')
flags.mark_flag_as_required('job')

## directories
flags.DEFINE_string('src_train_data','data/ratings_train.txt','train data')
flags.DEFINE_string('src_valid_data','data/ratings_test.txt','valid data')
flags.DEFINE_string('src_char2id','data/char2id.txt','char 2 id map')

flags.DEFINE_string('checkpoint_dir','checkpoints','dir to save model')

## model params
flags.DEFINE_integer('num_class', 2, 'number of classification class')
flags.DEFINE_integer('max_seq_len', 150, 'maximum char length of sentence')
flags.DEFINE_integer('lstm_hidden', 100, 'hidden size of lstm cell')
flags.DEFINE_integer('lstm_stack', 2, 'stack size of lstm')
flags.DEFINE_integer('num_char', 2000, 'number of characters to use')
flags.DEFINE_integer('embedding_size', 100, 'size of character embedding')

flags.DEFINE_integer('att_vec_size', 50, 'size of attention vector')
flags.DEFINE_integer('att_resolution', 20, 'resolution of attention')


## train params
flags.DEFINE_integer('batch_size', 32, 'size of mini batch')
flags.DEFINE_integer('num_epoch', 2, 'number of classification class')
flags.DEFINE_integer('prefetch_size', 1024, 'num prefetch')
flags.DEFINE_integer('save_every', 100, 'checkpoint save freq')

def main(argv):
    """ select process to do """

    if FLAGS.job == 'train':
        from src.trainer import Trainer
        Trainer().train()


if __name__ == "__main__":
    app.run(main)
    

