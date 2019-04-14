import os
import math
import collections
import tensorflow as tf
import numpy as np
from tensorflow import keras

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))

def swish(x):
    return x * tf.sigmoid(x)
 
class namespace():
    pass
args = namespace()
args.n_ctx = 512
args.n_embd = 768
args.n_head = 12
args.n_layer = 12
args.embd_pdrop = 0.1
args.attn_pdrop = 0.1
args.resid_pdrop = 0.1
args.clf_pdrop = 0.1
args.l2 = 0.1
args.n_transfer = 12
args.lm_coef = 0.5
args.b1 = 0.9
args.b2 = 0.999
args.e = 1e-8
args.n_valid = 374
args.afn = gelu

zeros_init = keras.initializers.Zeros()
ones_init = keras.initializers.Ones()

class LayerNorm(keras.Model):
    """Construct a layernorm module in the OpenAI style (epsilon inside the square root)."""

    def __init__(self, n_state=768, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = self.add_weight(shape=[n_state], initializer=ones_init)
        self.b = self.add_weight(shape=[n_state], initializer=zeros_init)
        self.e = e
    
    def call(self, x):
        u = tf.reduce_mean(x, -1, keepdims=True)
        s = tf.reduce_mean(tf.pow(x-u, 2), -1, keepdims=True)
        x = (x-u) / tf.sqrt(s+self.e)
        return self.g * x + self.b


class Conv1D(keras.Model):

    def __init__(self, nf=768*3, rf=1, nx=768):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1: # faster 1x1 conv
            self.w = self.add_weight(shape=[nx,nf], initializer=keras.initializers.RandomNormal(stddev=0.02))
            self.b = self.add_weight(shape=[nf], initializer=zeros_init)
        else:
            raise NotImplementedError

    def call(self, x):
        if self.rf == 1:
            size_out = list(x.shape[:-1]) + [self.nf]
            x = tf.matmul(tf.reshape(x, [-1, x.shape[-1]]), self.w) + self.b
            x = tf.reshape(x, size_out)
        else:
            raise NotImplementedError
        return x

    
class Attention(keras.Model):
    
    def __init__(self, nx=768, n_ctx=512, cfg=args, scale=False):
        super(Attention, self).__init__()
        n_state = nx # in Attention: n_state = 768 (nx=n_emb)
         # [switch nx => n_state from Block to Attention to keep identical to openai implem]
        assert n_state % cfg.n_head == 0
        self.b = self.add_weight(shape=[1, 1, n_ctx, n_ctx], initializer=ones_init) # register buffer
        self.b.assign(tf.linalg.LinearOperatorLowerTriangular(self.b).to_dense())
        self.n_head = cfg.n_head
        #self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state*3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = keras.layers.Dropout(cfg.attn_pdrop)
        self.resid_dropout = keras.layers.Dropout(cfg.resid_pdrop)
    
    def _attn(self, q, k, v):
        w = tf.matmul(q, k)
        if self.scale:
            w = w / tf.sqrt(tf.cast(v.shape[-1], tf.float32))
        # self.b may be larger than w, so we need to crop it
        b = self.b[:, :, :w.shape[-2], :w.shape[-1]]
        w = w * b + 1e-9 * (1 - b)
        w = tf.nn.softmax(w, -1)
        return tf.matmul(w, v)
    
    def merge_heads(self, x):
        x = tf.transpose(x, [0,2,1,3])
        new_x_shape = list(x.shape[:-2]) + [x.shape[-2]*x.shape[-1]]
        return tf.reshape(x, new_x_shape) # in openai implem: fct merge_states
    
    def split_heads(self, x, k=False):
        new_x_shape = list(x.shape[:-1]) + [self.n_head, x.shape[-1]//self.n_head]
        x = tf.reshape(x, new_x_shape) # in openai implem: fct split_states
        if k:
            return tf.transpose(x, [0,2,3,1])
        else:
            return tf.transpose(x, [0,2,1,3])
    
    def call(self, x):
        x = self.c_attn(x)
        query, key, value = tf.split(x, 3, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(keras.Model):

    def __init__(self, n_state=3072, cfg=args): # n_state=3072 (4*n_embd)
        super(MLP, self).__init__()
        nx = cfg.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = cfg.afn
        self.dropout = keras.layers.Dropout(cfg.resid_pdrop)
    
    def call(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(keras.Model):

    def __init__(self, n_ctx=512, cfg=args, scale=False):
        super(Block, self).__init__()
        nx = cfg.n_embd
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)
    
    def call(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class TransformerModel(keras.Model):
    """ Transformer model """

    def __init__(self, cfg=args, vocab=40558, n_ctx=512):
        super(TransformerModel, self).__init__()
        self.vocab = vocab
        self.embed = keras.layers.Embedding(vocab, cfg.n_embd)
        self.embed.build([1])
        self.drop = keras.layers.Dropout(cfg.embd_pdrop)
        self.h = [Block(n_ctx, cfg, scale=True) for _ in range(cfg.n_layer)]

    def call(self, x):
        x = tf.reshape(x, [-1,x.shape[-2],x.shape[-1]])
        e = self.drop(self.embed(x))
        # add the position information to input embeddings
        h = tf.reduce_sum(e, 2)
        for block in self.h:
            h = block(h)
        return h


class LMHead(keras.Model):
    """ Language Model Head for the transformer """

    def __init__(self, model, cfg=args, trunc_and_reshape=True):
        super(LMHead, self).__init__()
        self.n_embd = cfg.n_embd
        embed_shape = model.embed.weights[0].shape
        self.embed = model.embed.weights[0]
        self.decoder = lambda x: tf.matmul(x, tf.transpose(self.embed))
        self.trunc_and_reshape = trunc_and_reshape  # XD

    def call(self, h):
        # Truncated Language modeling logits (we remove the last token)
        h_trunc = tf.reshape(h[:, :-1], [-1, self.n_embd]) \
            if self.trunc_and_reshape else h  # XD
        lm_logits = self.decoder(h_trunc)
        return lm_logits


class MultipleChoiceHead(keras.Model):
    """ Classifier Head for the transformer """

    def __init__(self, clf_token=40480, cfg=args):
        super(MultipleChoiceHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.n_ctx = cfg.n_ctx
        self.clf_token = clf_token
        self.dropout = keras.layers.Dropout(cfg.clf_pdrop, [1, 2, cfg.n_embd, 1]) # might need to change the 1s to smth else
        self.linear = keras.layers.Dense(1, input_shape=[cfg.n_embd], 
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.02), 
            bias_initializer=keras.initializers.RandomNormal(stddev=1))
        self.linear.build([cfg.n_embd])

    def call(self, h, x):
        # Classification logits
        clf_h = tf.reshape(h, [-1, self.n_embd])
        flat = tf.reshape(x[..., 0], [-1])
        clf_h = tf.boolean_mask(clf_h, tf.equal(flat, self.clf_token))
        clf_h = tf.reshape(clf_h, [-1, x.shape[1], self.n_embd, 1])
        # This double transposition is there to replicate the behavior
        # of the noise_shape argument in the tensorflow
        # implementation.  For more details, see
        # https://github.com/huggingface/pytorch-openai-transformer-lm/issues/11
        # clf_h = self.dropout(clf_h.transpose(1, 2)).transpose(1, 2)
        clf_h = self.dropout(clf_h)
        clf_h = tf.reshape(clf_h, [-1, self.n_embd])
        clf_logits = self.linear(clf_h)

        return tf.reshape(clf_logits, [-1, x.shape[1]])


class ClfHead(keras.Model):
    """Classification Head for the transformer

    TODO: test this class."""
    def __init__(self, clf_token=40480, cfg=args, n_class=10):
        super(ClfHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = keras.layers.Dropout(cfg.clf_pdrop)
        self.linear = keras.layers.Dense(n_class, input_shape=[cfg.n_embd], 
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.02), 
            bias_initializer=keras.initializers.RandomNormal(stddev=1))

    def call(self, h, x):
        clf_h = tf.reshape(h, [-1, self.n_embd])
        flat = tf.reshape(x[..., 0], [-1])
        clf_h = clf_h[flat == self.clf_token, :]
        clf_h = tf.boolean_mask(clf_h, tf.equal(flat, self.clf_token))
        clf_h = self.dropout(clf_h)
        clf_logits = self.linear(clf_h)

        return clf_logits


class SimilarityHead(keras.Model):
    """ Similarity Head for the transformer

        TODO: test this class."""
    def __init__(self, clf_token=40480, cfg=args):
        super(SimilarityHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = keras.layers.Dropout(cfg.clf_pdrop)
        self.linear = keras.layers.Dense(n_class, input_shape=[cfg.n_embd], 
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.02), 
            bias_initializer=keras.initializers.RandomNormal(stddev=1))

    def call(self, h, x):
        sim_h = tf.reshape(h, [-1, self.n_embd])
        flat = tf.reshape(x[..., 0], [-1])
        sim_h = tf.boolean_mask(sim_h, tf.equal(flat, self.clf_token))
        sim_h = self.dropout(sim_h)
        sim_h = tf.reduce_sum(sim_h, 1)
        sim_logits = self.linear(sim_h)

        return sim_logits


class LMModel(keras.Model):
    """ Transformer with language model head only """
    def __init__(self, cfg=args, vocab=40990, n_ctx=512, return_probs=False):
        super(LMModel, self).__init__()
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.lm_head = LMHead(self.transformer, cfg, trunc_and_reshape=False)
        self.return_probs = return_probs
        if self.return_probs:
            pos_emb_mask = tf.zeros([1, 1, vocab]) # register buffer
            pos_emb_mask[:, :, -n_ctx:] = -1e12

    def call(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        if self.return_probs:
            lm_logits = tf.nn.softmax(lm_logits + self.pos_emb_mask, -1)
        return lm_logits


class DoubleHeadModel(keras.Model):
    """ Transformer with language model and task specific heads """
    def __init__(self, cfg=args, clf_token=40480, task_head_type='multiple_choice', vocab=40990, n_ctx=512):
        super(DoubleHeadModel, self).__init__()
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.lm_head = LMHead(self.transformer, cfg)
        if isinstance(task_head_type, str):
            if task_head_type == 'multiple_choice':
                self.task_head = MultipleChoiceHead(clf_token, cfg)
            elif task_head_type == 'similarity':
                self.task_head = SimilarityHead(clf_token, cfg)
            elif task_head_type == 'inference':
                # the three classes correspond to entailment, contradiction and neutral.
                self.task_head = ClfHead(clf_token, cfg, 3)
            else:
                raise ValueError("task_head_type is expected to be 'multiple_choice' "
                                 "'similarity', 'inference' or ('classification', n_class) "
                                 f"got {task_head_type}.")
        elif isinstance(task_head_type, collections.abc.Sequence) and len(task_head_type) == 2 and \
             task_head_type[0] == 'classification':
            n_class = task_head_type[1]
            self.task_head = ClfHead(clf_token, cfg, n_class)
        else:
            raise ValueError("task_head_type is expected to be 'multiple_choice' "
                             "'similarity', 'inference' or ('classification', n_class) "
                             f"got {task_head_type}.")

    def call(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        task_logits = self.task_head(h, x)

        return lm_logits, task_logits