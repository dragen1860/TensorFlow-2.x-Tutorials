import  tensorflow as tf

import  time
import  numpy as np
import  matplotlib.pyplot as plt
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from    tokenizer import get_tokenizer
from    transformer import Transformer
from    utils import CustomSchedule, create_masks
from    test import Translator




BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 128

train_dataset, val_dataset, tokenizer_en, tokenizer_zh = \
    get_tokenizer(MAX_SEQ_LENGTH, BATCH_SIZE)

# Chinese -> English translation
input_vocab_size = 21128
target_vocab_size = tokenizer_en.vocab_size + 2
dropout_rate = 0.1
num_layers=4
d_model=512
dff=2048
num_heads=8

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)

inp = tf.random.uniform((BATCH_SIZE, MAX_SEQ_LENGTH))
tar_inp = tf.random.uniform((BATCH_SIZE, MAX_SEQ_LENGTH))

fn_out, _ = transformer(inp, tar_inp,
                        True,
                        enc_padding_mask=None,
                        look_ahead_mask=None,
                        dec_padding_mask=None)
print(tar_inp.shape)  # (batch_size, tar_seq_len)
print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
transformer.summary()



learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)



loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                    name='train_accuracy')


checkpoint_path = "./zh-en/transformer"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')



@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


# Chinese is used as the input language and English is the target language.
translator = Translator(tokenizer_zh, tokenizer_en, transformer, MAX_SEQ_LENGTH)

for epoch in range(20):

    (cn_code, en_code) = next(iter(val_dataset))
    cn_code, en_code = cn_code[epoch].numpy(), en_code[epoch].numpy()
    # print(cn_code)
    # print(en_code)
    en = tokenizer_en.decode([i for i in en_code if i < tokenizer_en.vocab_size])
    cn_code = [int(i)
                    for i in cn_code if (i!=101 and i!=102 and i!=1 and i!=0)]
    # print(cn_code)
    cn = tokenizer_zh.convert_ids_to_tokens(cn_code)
    cn = "".join(cn)
    translator.do(cn)
    print('Real:', en)
    print('\n')



    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> chinese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 3 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))



