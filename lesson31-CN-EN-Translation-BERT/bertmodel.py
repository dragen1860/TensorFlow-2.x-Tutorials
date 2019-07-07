import tensorflow as tf

from bert import BertModelLayer
from bert.loader import StockBertConfig, load_stock_weights
from bert.loader import map_to_stock_variable_name


from    utils import positional_encoding
from    attlayer import DecoderLayer

class Config(object):
    def __init__(self, num_layers, d_model, dff, num_heads):
        self.num_layers = num_layers
        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads


# In[ ]:

def build_encoder(config_file):
    with tf.io.gfile.GFile(config_file, "r") as reader:
        stock_params = StockBertConfig.from_json_string(reader.read())
        bert_params = stock_params.to_bert_model_layer_params()

    return BertModelLayer.from_params(bert_params, name="bert")



class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights



class Transformer(tf.keras.Model):
    def __init__(self, config,
                 target_vocab_size,
                 bert_config_file,
                 bert_training=False,
                 rate=0.1,
                 name='transformer'):
        super(Transformer, self).__init__(name=name)

        self.encoder = build_encoder(config_file=bert_config_file)
        self.encoder.trainable = bert_training

        self.decoder = Decoder(config.num_layers, config.d_model,
                               config.num_heads, config.dff, target_vocab_size, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def load_stock_weights(self, bert: BertModelLayer, ckpt_file):
        assert isinstance(bert, BertModelLayer), "Expecting a BertModelLayer instance as first argument"
        assert tf.compat.v1.train.checkpoint_exists(ckpt_file), "Checkpoint does not exist: {}".format(ckpt_file)
        ckpt_reader = tf.train.load_checkpoint(ckpt_file)

        bert_prefix = 'transformer/bert'

        weights = []
        for weight in bert.weights:
            stock_name = map_to_stock_variable_name(weight.name, bert_prefix)
            if ckpt_reader.has_tensor(stock_name):
                value = ckpt_reader.get_tensor(stock_name)
                weights.append(value)
            else:
                raise ValueError("No value for:[{}], i.e.:[{}] in:[{}]".format(
                    weight.name, stock_name, ckpt_file))
        bert.set_weights(weights)
        print("Done loading {} BERT weights from: {} into {} (prefix:{})".format(
            len(weights), ckpt_file, bert, bert_prefix))

    def restore_encoder(self, bert_ckpt_file):
        # loading the original pre-trained weights into the BERT layer:
        self.load_stock_weights(self.encoder, bert_ckpt_file)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training=self.encoder.trainable)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

