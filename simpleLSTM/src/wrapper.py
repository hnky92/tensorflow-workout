import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os,sys

basedir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(basedir)

from generator import ManageData


class Wrapper(object):

    def __init__(self, src_model, src_char2id, max_len):
        """initializer preprocess module, make session, load model"""
        self.MD = ManageData(src_char2id, max_len)

        # make graph
        graph = tf.Graph()

        with graph.as_default():
            config = tf.ConfigProto(
                device_count = {'GPU':1}
                )

        # make session
        self.sess = sess = tf.Session()

        # restore checkpoint
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(src_model))
            saver.restore(sess, src_model)

            # input placeholder for charid, seq len
            self.X = sess.graph.get_operation_by_name('X').outputs[0]
            self.X_len = sess.graph.get_operation_by_name('length').outputs[0]

            # output: prediction
            self._pred = sess.graph.get_operation_by_name('pred').outputs[0]


    def preprocess(self, sentences):
        """ create batched feature dictionary"""
        features = {"sentence":[],"X":[],"seq_len":[]}

        for sent in sentences:
             _feat_dict = self.MD.preprocess(sent)

             # parse result
             features["sentence"].append(_feat_dict["sentence"])
             features["X"].append(_feat_dict["X"])
             features["seq_len"].append(_feat_dict["sequence_length"])

        return features

    def run_batch(self, sentences):
        """ forward sentence to model, return prediction result """
        features = self.preprocess(sentences)
        print(features)

        list_pred = self.sess.run([self._pred],
                                  feed_dict={self.X:features["X"],
                                             self.X_len:features["seq_len"]})

        return list_pred


if __name__ == "__main__":
    config = {
            "src_model":"checkpoints/1584974736/model-12400",
            "src_char2id":"data/char2id.txt",
            "max_len":150
            }

    w = Wrapper(config["src_model"], config["src_char2id"], config["max_len"])
    res = w.run_batch(['매력적이지 않고, 스토리도 어디서 많이 본듯하다','한국에서 이런 영화가 나왔다는게 감사한 작품!'])

    print(res)
