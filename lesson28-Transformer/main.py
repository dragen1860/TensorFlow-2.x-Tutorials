import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

class namespace():
    pass
args = namespace()
args.n_ctx = 512
args.n_embd = 768
args.n_head = 12
args.n_lar = 12
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

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))

def swish(x):
    return x * tf.sigmoid(x)

zeros_init = keras.initializers.Zeros()
ones_init = keras.initializers.Ones()


class LayerNorm(keras.Model):

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
        self.b = self.add_weight(shape=[1, 1, n_ctx, n_ctx], initializer=ones_init)
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
            w = w / tf.sqrt(v.shape[-1])
        # self.b may be larger than w, so we need to crop it
        b = self.b[:, :, :w.shape[-2], :w.shape[-1]]
        w = w * b + 1e-9 * (1 - b)
        w = tf.nn.softmax(w, -1)
        return tf.matmul(w, v)
    
    def merge_heads(self, x):
        x = tf.transpose(x, [0,2,1,3])
        new_x_shape = list(x.shape[:-2]) + [x.shape[-2]*x.shape[1]]
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