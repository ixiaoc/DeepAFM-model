import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DeepAFM(BaseEstimator, TransformerMixin):

    def __init__(self, feature_size_one_hot, field_size_one_hot,   # one-hot feature parameters
                 feature_size_multi_value, field_size_multi_value,  # multi-value feature parameters
                 embedding_size=8, attention_size=10,  # AFM parameters
                 deep_layers=None, dropout_deep=None, deep_layer_activation=tf.nn.relu,  # DNN parameters
                 epoch=10, batch_size=1024, learning_rate=0.001, optimizer="adam",  # training parameters
                 use_afm=True, use_deep=True, random_seed=2018,  # random parameters
                 loss_type="mse", eval_metric=mean_squared_error, l2_reg=0.0,  # evaluating parameters
                 rnn_size=100, num_rnn_layers=1, keep_lstm=0.5, num_unroll_steps=200, field_size_text=3, # LSTM parameters
                 word_embeddings=None  # word vector parameters
                 ):

        self.feature_size = feature_size_one_hot + feature_size_multi_value
        self.field_size = field_size_one_hot + field_size_multi_value + field_size_text
        self.field_size_one_hot = field_size_one_hot
        self.field_size_multi_value = field_size_multi_value
        self.field_size_text = field_size_text

        self.embedding_size = embedding_size
        self.attention_size = attention_size

        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layer_activation

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.use_afm = use_afm
        self.use_deep = use_deep
        self.random_seed = random_seed

        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.l2_reg = l2_reg

        self.train_result, self.valid_result = [], []

        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.dropout_lstm = keep_lstm
        self.num_unroll_steps = num_unroll_steps  # sentence length

        self.word_embeddings = word_embeddings

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')  # label
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')
            self.dropout_keep_lstm = tf.placeholder(tf.float32, shape=None, name='dropout_deep_lstm')

            # one-hot feature part
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index_one_hot')
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value_one_hot')

            # multi_value feature part
            self.feat_index_m = tf.placeholder(tf.int32, shape=[self.field_size_multi_value, None, None], name='feat_index_multi_value')
            self.feat_value_m = tf.placeholder(tf.float32, shape=[self.field_size_multi_value, None, None], name='feat_value_multi_value')

            # text feature part
            self.text_data = tf.placeholder(tf.int32, [None, self.field_size_text, self.num_unroll_steps])  # N * Ft * S
            self.mask_x = tf.placeholder(tf.float32, [None, self.field_size_text, self.num_unroll_steps])  # S * Ft * N

            self.weights = self._initialize_weights()

            # Embeddings

            # one-hot feature
            self.embeddings_one_hot = tf.nn.embedding_lookup(self.weights['feature_embeddings_one_hot'], self.feat_index)  # N * Fo * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size_one_hot, 1])
            self.embeddings_one_hot = tf.multiply(self.embeddings_one_hot, feat_value)  # N * Fo * K

            # multi_value feature
            embeddings_text_list = []
            for i in range(self.field_size_multi_value):
                embeddings_multi_value = tf.nn.embedding_lookup(self.weights['feature_embeddings_multi_value'][i], self.feat_index_m[i, :, :])  # N * Fmi * K
                feat_value_m = tf.reshape(self.feat_value_m[i, :, :], shape=[-1, self.field_size_one_hot, 1])
                embeddings_multi_value = tf.multiply(embeddings_multi_value, feat_value_m)  # N * Fmi * K
                embeddings_multi_value = tf.reduce_sum(embeddings_multi_value, axis=1)  # N * K


            # text feature
            self.word_embeddings = tf.Variable(tf.to_float(self.word_embeddings), trainable=True, name="word_embeddings")  # 字典长度 * E
            embeddings_text_list = []
            for i in range(self.field_size_text):
                embeddings_text_list.append(self.bilstm_network(self.text_data[:, i], self.mask_x[:, i], i))  # N * K
            self.embeddings_text = tf.stack(embeddings_text_list)  # Ft * N * K
            self.embeddings_text = tf.transpose(self.embeddings_text, perm=[1, 0, 2])  # N * Ft * K

            # concat feature
            self.embeddings = tf.concat([self.embeddings_one_hot, self.embeddings_text], axis=1)  # N * F * K

            # AFM component
            # element_wise
            element_wise_product_list = []
            for i in range(self.field_size):
                for j in range(i + 1, self.field_size):
                    element_wise_product_list.append(tf.multiply(self.embeddings[:, i, :], self.embeddings[:, j, :]))  # N * K

            self.element_wise_product = tf.stack(element_wise_product_list)  # [F * (F - 1)/2] * N * K
            self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0, 2], name='element_wise_product')  # N * [F * (F - 1)/2] *  K

            # attention part
            num_interactions = int(self.field_size * (self.field_size - 1) / 2)
            # wx+b -> relu(wx+b) -> h*relu(wx+b)
            self.attention_wx_plus_b = tf.reshape(tf.add(tf.matmul(tf.reshape(self.element_wise_product, shape=(-1, self.embedding_size)), self.weights['attention_w']),
                       self.weights['attention_b']),
                shape=[-1, num_interactions, self.attention_size])  # N * [F * (F - 1)/2] * A

            self.attention_exp = tf.exp(tf.reduce_sum(tf.multiply(tf.nn.relu(self.attention_wx_plus_b),
                                                                  self.weights['attention_h']),
                                                      axis=2, keepdims=True))  # N * [F * (F - 1) / 2] * 1

            self.attention_exp_sum = tf.reduce_sum(self.attention_exp, axis=1, keepdims=True)  # N * 1 * 1

            self.attention_out = tf.div(self.attention_exp, self.attention_exp_sum, name='attention_out')  # N * [F * (F - 1)/2] * 1

            self.attention_x_product = tf.reduce_sum(tf.multiply(self.attention_out, self.element_wise_product), axis=1, name='afm')  # N * K

            self.attention_part_sum = tf.matmul(self.attention_x_product, self.weights['attention_p'])  # N * 1

            # bias
            self.y_bias = self.weights['bias'] * tf.ones_like(self.label)  # N * 1

            # out
            self.out_afm = tf.add_n([self.attention_part_sum, self.y_bias], name='out_afm')  # N * 1

            print(self.out_afm)

            # Deep component
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])  # N * D

            # out
            self.out_deep = tf.add(tf.matmul(self.y_deep, self.weights['deep_projection']), self.weights['deep_bias'])  # N * 1
            print(self.out_deep)

            # concat output
            concat_input = tf.concat([self.out_afm, self.out_deep], axis=1)
            self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])  # N * 1

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # # l2 regularization on weights
            # if self.l2_reg > 0:
            #     self.loss += tf.contrib.layers.l2_regularizer(
            #         self.l2_reg)(self.weights["concat_projection"])
            #     self.loss += tf.contrib.layers.l2_regularizer(
            #         self.l2_reg)(self.weights["deep_projection"])
            #     for i in range(len(self.deep_layers)):
            #         self.loss += tf.contrib.layers.l2_regularizer(
            #             self.l2_reg)(self.weights["layer_%d" % i])

            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name='feature_embeddings')
        weights['bias'] = tf.Variable(tf.constant(0.1), name='bias')

        # attention part
        glorot = np.sqrt(2.0 / (self.attention_size + self.embedding_size))

        weights['attention_w'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.embedding_size, self.attention_size)),
                                             dtype=tf.float32, name='attention_w')

        weights['attention_b'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.attention_size,)),
                                             dtype=tf.float32, name='attention_b')

        weights['attention_h'] = tf.Variable(np.random.normal(loc=0, scale=1, size=(self.attention_size,)),
                                             dtype=tf.float32, name='attention_h')

        weights['attention_p'] = tf.Variable(np.ones((self.embedding_size, 1)), dtype=np.float32)

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))

        weights['layer_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32
        )
        weights['bias_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32
        )

        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        weights['deep_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot,
                                                                  size=(self.deep_layers[-1], 1)), dtype=np.float32)
        weights['deep_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        # final concat projection layer
        glorot = np.sqrt(2.0 / 3)
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(2, 1)),
                                                   dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    def get_batch(self, Xi, Xv, y, Xt, Xm, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], Xt[start:end], Xm[start:end], [[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c, d, e):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(e)

    def predict(self, Xi, Xv, Xt, Xm, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.text_data: Xt,
                     self.mask_x: Xm,
                     self.label: y,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                     self.dropout_keep_lstm: 1.0
                     }

        loss = self.sess.run([self.loss], feed_dict=feed_dict)

        return loss

    def fit_on_batch(self, Xi, Xv, Xt, Xm, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.text_data: Xt,
                     self.mask_x: Xm,
                     self.label: y,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.dropout_keep_lstm: self.dropout_lstm
                     }  # TODO：self.dropout_lstm

        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

        return loss

    def fit(self, Xi_train, Xv_train, y_train, Xt_train, Xm_train,
            Xi_valid=None, Xv_valid=None, y_valid=None, Xt_train_valid=None, Xm_train_valid=None):

        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train, Xt_train, Xm_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch, Xt_batch, Xm_batch = self.get_batch(Xi_train, Xv_train, y_train, Xt_train, Xm_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch, Xt_batch, Xm_batch)

            if has_valid:
                y_valid = np.array(y_valid).reshape((-1, 1))
                loss = self.predict(Xi_valid, Xv_valid, Xt_train_valid, Xm_train_valid, y_valid)
                print("epoch", epoch, "loss", loss)

    def bilstm_network(self, input_data, mask_x, number):

        with tf.variable_scope(str(number)):
            # build BILSTM network
            # forward rnn
            fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size)  # R
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell] * self.num_rnn_layers, state_is_tuple=True)  # R
            # backforward rnn
            bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size)  # R
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell] * self.num_rnn_layers, state_is_tuple=True)  # R

            # embedding layer
            inputs = tf.nn.embedding_lookup(self.word_embeddings, input_data)  # N * S * E
            print(inputs)
            inputs = tf.nn.dropout(inputs, self.dropout_keep_lstm)  # N * S * E

            inputs = [tf.squeeze(input, [1]) for input in tf.split(inputs, self.num_unroll_steps, 1)]  # S * N * E

            out_put, _, _ = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, inputs,
                                                           dtype=tf.float32)  # S * N * (R * 2)
            out_put = tf.transpose(out_put, perm=[1, 0, 2])  # N * S * (R * 2)

            with tf.variable_scope("text_%d" % number):
                out_put = attention(out_put, self.attention_size)  # N * (R * 2)

                out_put = tf.nn.dropout(out_put, self.dropout_keep_lstm)  # N * (R * 2)
                print(111)
                print(out_put)

                # TODO: self.embedding_size 修改为 config.embedding_size
                w = tf.get_variable("lstm_embedding_w",
                                    initializer=tf.random_normal([self.rnn_size * 2, self.embedding_size],
                                                                 stddev=0.1))
                b = tf.get_variable("lstm_embedding_b",
                                    initializer=tf.random_normal([self.embedding_size], stddev=0.1))
                out_put = tf.add(tf.matmul(out_put, w), b)  # N * K
                out_put = self.deep_layers_activation(out_put)  # N * K
                out_put = tf.nn.dropout(out_put, self.dropout_keep_lstm)  # N * K
                print(out_put)

            return out_put


def attention(inputs, attention_size):
    """
    Attention mechanism layer.
    :param inputs: outputs of RNN/Bi-RNN layer (not final state)
    :param attention_size: linear size of attention weights
    :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
    """
    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    # TODO: lstm_attention维度问题
    if isinstance(inputs, tuple):
        inputs = tf.concat(2, inputs)

    sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.get_variable("W_omega", initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.get_variable("b_omega", initializer=tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.get_variable("u_omega", initializer=tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    return output
