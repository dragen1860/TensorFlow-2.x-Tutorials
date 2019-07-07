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


from utils import create_masks


class Translator:

    def __init__(self, tokenizer_zh, tokenize_en, model, MAX_SEQ_LENGTH):

        self.tokenizer_zh = tokenizer_zh
        self.tokenizer_en = tokenize_en
        self.model = model
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH


    def encode_zh(self, zh):
        tokens_zh = self.tokenizer_zh.tokenize(zh)
        lang1 = self.tokenizer_zh.convert_tokens_to_ids(['[CLS]'] + tokens_zh + ['[SEP]'])

        return lang1





    def evaluate(self, inp_sentence):
        # normalize input sentence
        inp_sentence = self.encode_zh(inp_sentence)
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.tokenizer_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(self.MAX_SEQ_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.model(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self.tokenizer_en.vocab_size + 1):
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights





    def plot_attention_weights(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))

        sentence_ids = self.encode_zh(sentence)

        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap='viridis')

            fontdict = {'fontsize': 10, 'family': 'DFKai-SB'}

            ax.set_xticks(range(len(sentence_ids)))
            ax.set_yticks(range(len(result)))

            ax.set_ylim(len(result) - 1.5, -0.5)

            ax.set_xticklabels(
                self.tokenizer_zh.convert_ids_to_tokens(sentence_ids),
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([self.tokenizer_en.decode([i]) for i in result
                                if i < self.tokenizer_en.vocab_size],
                               fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head + 1))

        plt.tight_layout()
        plt.show()


# In[ ]:


    def do(self, sentence, plot=''):
        result, attention_weights = self.evaluate(sentence)

        predicted_sentence = self.tokenizer_en.decode([i for i in result
                                                  if i < self.tokenizer_en.vocab_size])

        print('Chinese src: {}'.format(sentence))
        print('Translated : {}'.format(predicted_sentence))

        if plot:
            self.plot_attention_weights(attention_weights, sentence, result, plot)




def main():
    # In[42]:

    sentence_ids = encode_zh("我爱你啊")
    print(tokenizer_zh.convert_ids_to_tokens(sentence_ids))

    # In[ ]:

    # In[51]:

    translate(transformer, '虽然继承了祖荫，但朴槿惠已经证明了自己是个机敏而老练的政治家——她历经20年才爬上韩国大国家党最高领导层并成为全国知名人物。')
    print(
        'Real translation: While Park derives some of her power from her family pedigree, she has proven to be an astute and seasoned politician –&nbsp;one who climbed the Grand National Party’s leadership ladder over the last two decades to emerge as a national figure.')

    # In[59]:

    translate(transformer, "我爱你是一件幸福的事情。")

    # ## Save weights

    # In[ ]:

    transformer.save_weights('bert_nmt_ckpt')

    # In[49]:

    new_transformer = Transformer(config=config,
                                  target_vocab_size=target_vocab_size,
                                  bert_config_file=bert_config_file)

    fn_out, _ = new_transformer(inp, tar_inp,
                                True,
                                look_ahead_mask=None,
                                dec_padding_mask=None)
    new_transformer.load_weights('bert_nmt_ckpt')

    translate(new_transformer, '我爱你')

if __name__ == '__main__':
    main()