# use tf v1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from absl import flags

FLAGS = flags.FLAGS


class SimpleLSTM():
    def __init__(self):
        self.num_class= FLAGS.num_class
        self.initializer = tf.initializers.glorot_normal()

        self.X = tf.placeholder(tf.int32, shape=[None, FLAGS.max_seq_len], name='X')
        self.Y = tf.placeholder(tf.int32, shape=[None], name='Y')
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name='length')

        self._build()


    def create_cell(self, stack_size, scope):
        """ CREATE CELL """
        with tf.name_scope(scope):
            cells = []
            for _ in range(stack_size):

                cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.lstm_hidden, 
                                               state_is_tuple=True)
                cells.append(cell)

            cells = tf.nn.rnn_cell.MultiRNNCell(cells)

        return cell


    ## create bidi stack
    def create_bidilstm(self, _cell_fw, _cell_bw, X, X_len, scope):
        """ make bidi dynamic lstm"""
        with tf.name_scope(scope):
            ## use cells as bidi-dynamic
            output, _  = tf.nn.bidirectional_dynamic_rnn(
                                                      cell_fw = _cell_fw,
                                                      cell_bw = _cell_bw,
                                                      inputs = X,
                                                      dtype = tf.float32,
                                                      sequence_length = X_len,
                                                      time_major = False,
                                                      scope = scope)
            ## add forward backward
            output = tf.concat((output[0],output[1]),2)

        return output


    def build_lstm(self, X, X_len, scope):
        """ BUILD LSTM """
        ## f,b cell 
        cell_fw = self.create_cell(FLAGS.lstm_stack, scope+'_forward')
        cell_bw = self.create_cell(FLAGS.lstm_stack, scope+'_backward')

        ## create stack 
        output = self.create_bidilstm(cell_fw, cell_bw, X, X_len, scope+'_lstm')

        return output


    def build_embedding(self,X):
        """ create character embedding layer """
        embedding = tf.get_variable("char_embedding",
                                    [FLAGS.num_char,FLAGS.embedding_size],
                                    initializer=self.initializer,
                                    dtype=tf.float32)

        embedded = tf.nn.embedding_lookup(embedding,X)
        return embedded


    def self_attention(self,H):
        """ self attentive layer"""
        d = FLAGS.embedding_size
        d_a = FLAGS.att_vec_size
        u = FLAGS.lstm_hidden
        r = FLAGS.att_resolution

        self.H = H
        tf.identity(H, 'H')

        initializer = self.initializer

        self.W_s1 = tf.get_variable('W_s1', shape = [d_a, 2 * u],
                    initializer = initializer)
        self.W_s2 = tf.get_variable('W_s2', shape = [r, d_a],
                    initializer = initializer)

        # Calculate attention matrix
        # A : Attention matrix of self-attntion
        self.A = A = tf.nn.softmax(
          tf.map_fn(
            lambda x : tf.matmul(self.W_s2, x),
            tf.tanh(
              tf.map_fn(
                lambda x : tf.matmul(self.W_s1, tf.transpose(x)),
                self.H
              )
            )
          )
        )
        tf.identity(self.A, 'A')

        # M : Attntion applied result (= Output of self-attention)
        # A shape : (batch, r, seq_length)
        # H shape : (batch, seq_length, 2*u)
        # M shape : (batch, r, 2*u) -> weighted average of each dimension. in resolution 1~r
        self.M = tf.matmul(A, H)

        self.M = tf.reshape(self.M,[-1,2*r*u])
        tf.identity(self.M, 'M')
        # return shape : (batch, r*2*u)
        return self.M


    def calc_loss(self, logit, Y):
        """ get categorical cross entropy loss btw input_logit and target
        """

        y_one_hot = tf.one_hot(Y,depth=self.num_class)

        log_probs = tf.nn.log_softmax(logit, axis=-1, name='log_probs')

        per_example_loss = -tf.reduce_sum(log_probs * y_one_hot, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return loss, log_probs


    def _build(self):
        """ build model """
        # embedding lookup for each character
        X_embedded = self.build_embedding(self.X)

        # connects to lstm
        lstm_out = self.build_lstm(X_embedded, self.seq_len, 'bi_lstm')

        # apply attention
        logit = self.self_attention(lstm_out)

        # calc loss
        logit = tf.layers.dense(logit, self.num_class, activation=None, name='logit_out') 
        self.loss, self.log_probs = self.calc_loss(logit,self.Y)

        # prediction
        self.pred = tf.argmax(self.log_probs, axis=1, name='pred')

        # acc
        correct_pred = tf.equal(tf.cast(self.pred,tf.int32),self.Y)
        self.acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

        # summary
        loss_summary = tf.summary.scalar("loss",self.loss)
        acc_summary = tf.summary.scalar("acc",self.acc)

        self.summary_op = tf.summary.merge([loss_summary,acc_summary])

        # use Adam optimizer
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train = tf.train.AdamOptimizer().minimize(self.loss,global_step=self.global_step)




