
# ## Evaluate

# The following steps are used for evaluation:
#
# - Encode the input sentence using the Portuguese tokenizer (tokenizer_pt). Moreover, add the start and end token so the input is equivalent to what the model is trained with. This is the encoder input.
#
# - The decoder input is the start token == tokenizer_en.vocab_size.
#
# - Calculate the padding masks and the look ahead masks.
#
# - The decoder then outputs the predictions by looking at the encoder output and its own output (self-attention).
#
# - Select the last word and calculate the argmax of that.
#
# - Concatentate the predicted word to the decoder input as pass it to the decoder.
#
# - In this approach, the decoder predicts the next word based on the previous words it predicted.

# In[ ]:


def encode_zh(zh):
    tokens_zh = tokenizer_zh.tokenize(zh)
    lang1 = tokenizer_zh.convert_tokens_to_ids(['[CLS]'] + tokens_zh + ['[SEP]'])

    return lang1


# In[42]:


sentence_ids = encode_zh("我爱你啊")
print(tokenizer_zh.convert_ids_to_tokens(sentence_ids))


# In[ ]:


def evaluate(transformer, inp_sentence):
    # normalize input sentence
    inp_sentence = encode_zh(inp_sentence)
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_SEQ_LENGTH):
        combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, tokenizer_en.vocab_size + 1):
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


# In[ ]:


def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence_ids = encode_zh(sentence)

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
            tokenizer_zh.convert_ids_to_tokens(sentence_ids),
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
                            if i < tokenizer_en.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


# In[ ]:


def translate(transformer, sentence, plot=''):
    result, attention_weights = evaluate(transformer, sentence)

    predicted_sentence = tokenizer_en.decode([i for i in result
                                              if i < tokenizer_en.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)


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

# ## Summary

# In this tutorial, you learned about positional encoding, multi-head attention, the importance of masking and how to create a transformer with BERT.
#
# Try using a different dataset to train the transformer. Futhermore, you can implement beam search to get better predictions.
#
# At last, you can copy pre-trained weights and relevant files to your drive.

# In[53]:


# 运行此单元格即可装载您的 Google 云端硬盘。
from google.colab import drive

drive.mount('/content/drive')

# In[ ]:


get_ipython().system('cp bert_nmt_ckpt* "./drive/My Drive/models/nmt/"')
get_ipython().system('cp vocab_en* "./drive/My Drive/models/nmt/"')
