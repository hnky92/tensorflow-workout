# use tf v1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import time, os

from src.generator import ManageData
from src.model import SimpleLSTM

from tqdm import tqdm
from absl import flags

FLAGS = flags.FLAGS

class Trainer(object): 
    def __init__(self):
        timestamp = str(int(time.time()))
        self.save_dir = os.path.join(FLAGS.checkpoint_dir,str(timestamp))

        # init model
        self.model = SimpleLSTM()

        # GPU ops
        graph = tf.Graph()
        with graph.as_default():
            config = tf.ConfigProto(
                device_count = {'GPU':1}
                )
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        # create session
        self.sess = sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        # summary writer
        train_summary_path = os.path.join(os.path.join(self.save_dir,"summaries","train"))
        valid_summary_path = os.path.join(os.path.join(self.save_dir,"summaries","valid"))

        self.saver = tf.train.Saver(max_to_keep=8)
        self.train_writer = tf.summary.FileWriter(train_summary_path,sess.graph)
        self.valid_writer = tf.summary.FileWriter(valid_summary_path)

        tf.gfile.MakeDirs(train_summary_path)
        tf.gfile.MakeDirs(valid_summary_path)


    def train_step(self, tr_generator, vl_iterator):
        """ train iteration """

        # update weight
        feed_element = self.sess.run(tr_generator)

        _, loss, acc, _summary = self.sess.run([self.model.train, 
                                                self.model.loss,
                                                self.model.acc, 
                                                self.model.summary_op],
                                                feed_dict={self.model.X:feed_element['X'],
                                                           self.model.Y:feed_element['Y'],
                                                           self.model.seq_len:feed_element['sequence_length']})

        current_step = tf.train.global_step(self.sess, self.model.global_step)
        self.train_writer.add_summary(_summary, global_step=current_step)

        # print log
        if current_step % 20 == 0:
            print('[step %d] loss: %f, acc: %f' % (current_step, loss, acc))

        # validation & save
        if current_step % FLAGS.save_every == 0:
            self.sess.run(vl_iterator.initializer)
            vl_generator = vl_iterator.get_next()

            acc_tmp = []
            loss_tmp = []
            while 1:
                try:
                    feed_element = self.sess.run(vl_generator)

                    loss, acc = self.sess.run([self.model.loss,
                                               self.model.acc],
                                               feed_dict={self.model.X:feed_element['X'],
                                                          self.model.Y:feed_element['Y'],
                                                          self.model.seq_len:feed_element['sequence_length']})

                    loss_tmp.append(loss)
                    acc_tmp.append(acc)

                except tf.errors.OutOfRangeError:
                    break

            avg_loss = sum(loss_tmp)/len(loss_tmp)
            avg_acc = sum(acc_tmp)/len(acc_tmp)
            print('[Validation] loss: %f, acc: %f' % (avg_loss, avg_acc))

            val_summary = tf.Summary()
            val_summary.value.add(tag="loss", simple_value=avg_loss)
            val_summary.value.add(tag="acc", simple_value=avg_acc)

            self.valid_writer.add_summary(val_summary, global_step=current_step)
            self.saver.save(self.sess, os.path.join(self.save_dir,'model'), global_step = current_step)


    def set_data(self, dir_data, batch_size, num_epoch=None):
        """ make tf iterator via generator"""
        print(dir_data)
        MD = ManageData(FLAGS.src_char2id, FLAGS.max_seq_len, [dir_data])

        # define shape and types
        output_shapes = {"sentence":[],
                         "X":[FLAGS.max_seq_len],
                         "Y":[],
                         "sequence_length":[]}

        output_types = {"sentence":tf.string, 
                        "X":tf.int32, 
                        "Y":tf.int32,
                        "sequence_length":tf.int32}

        # get generator
        tf_dataset = tf.data.Dataset.from_generator(MD.generator,
                                                    output_shapes=output_shapes,
                                                    output_types=output_types)

        # dataset settings
        tf_dataset = tf_dataset.batch(batch_size)
        tf_dataset = tf_dataset.shuffle(FLAGS.prefetch_size)
        tf_dataset = tf_dataset.prefetch(FLAGS.prefetch_size)
        tf_dataset = tf_dataset.repeat(num_epoch)

        # init iterator
        iterator = tf_dataset.make_initializable_iterator()

        return iterator


    def train(self):
        """ load train, valid data and iterate train step"""
        # load data
        tr_iterator = self.set_data(FLAGS.src_train_data, FLAGS.batch_size, FLAGS.num_epoch) 
        vl_iterator = self.set_data(FLAGS.src_valid_data, FLAGS.batch_size, 1) 

        self.sess.run(tr_iterator.initializer)
        tr_generator = tr_iterator.get_next()
        
        while 1:
            try:
                self.train_step(tr_generator, vl_iterator)

            except tf.errors.OutOfRangeError:
                print("end of training")
                break
