import  tensorflow as tf
from    tensorflow import keras
from    layers import *
from    metrics import *
from    config import args 





class MLP(keras.Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += args.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=args.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=args.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(keras.Model):

    def __init__(self, input_dim, output_dim, num_features_nonzero, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)

        self.layers_ = []
        self.layers_.append(GraphConvolution(input_dim=self.input_dim, # 1433
                                            output_dim=args.hidden1, # 16
                                            num_features_nonzero=num_features_nonzero,
                                            activation=tf.nn.relu,
                                            dropout=args.dropout,
                                            is_sparse_inputs=True))





        self.layers_.append(GraphConvolution(input_dim=args.hidden1, # 16
                                            output_dim=self.output_dim, # 7
                                            num_features_nonzero=num_features_nonzero,
                                            activation=lambda x: x,
                                            dropout=args.dropout))


        for p in self.trainable_variables:
            print(p.name, p.shape)

    def call(self, inputs, training=None):
        """

        :param inputs:
        :param training:
        :return:
        """
        x, label, mask, support = inputs

        outputs = [x]

        for layer in self.layers:
            hidden = layer((outputs[-1], support), training)
            outputs.append(hidden)
        output = outputs[-1]

        # # Weight decay loss
        loss = tf.zeros([])
        for var in self.layers_[0].trainable_variables:
            loss += args.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        loss += masked_softmax_cross_entropy(output, label, mask)

        acc = masked_accuracy(output, label, mask)

        return loss, acc



    def predict(self):
        return tf.nn.softmax(self.outputs)
