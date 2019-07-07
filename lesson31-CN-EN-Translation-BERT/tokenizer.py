import  tensorflow as tf
import  tensorflow_datasets as tfds


import  collections
import  unicodedata
import  os,sys
import  numpy as np





def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False




def get_tokenizer(MAX_SEQ_LENGTH, BATCH_SIZE):
    # ## Setup input pipleline

    # Use TFDS to load the wmt2019 zh-en translation dataset.

    if not os.path.exists('chinese_L-12_H-768_A-12'):
        # get_ipython().system('wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip')
        # get_ipython().system('unzip chinese_L-12_H-768_A-12')
        print('download pretrained first!')
        sys.exit()

    config = tfds.translate.wmt.WmtConfig(
        description="WMT 2019 translation task dataset.",
        version="0.0.3",
        language_pair=("zh", "en"),
        subsets={
            tfds.Split.TRAIN: ["newscommentary_v13"],
            tfds.Split.VALIDATION: ["newsdev2017"],
        }
    )

    builder = tfds.builder("wmt_translate", config=config)
    print(builder.info.splits)
    builder.download_and_prepare()
    datasets = builder.as_dataset(as_supervised=True)
    print('datasets is {}'.format(datasets))

    # In[ ]:

    train_examples = datasets['train']
    val_examples = datasets['validation']

    # In[ ]:

    for zh, en in train_examples.take(1):
        # print((zh))
        print(tf.compat.as_text(zh.numpy()))
        print(tf.compat.as_text(en.numpy()))

    # Create a custom subwords tokenizer from the training dataset for the decoder.

    # In[ ]:

    vocab_file = 'vocab_en'
    if os.path.isfile(vocab_file + '.subwords'):
        tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_file)
    else:
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for zh, en in train_examples), target_vocab_size=2 ** 13)
        tokenizer_en.save_to_file('vocab_en')

    sample_string = 'Transformer is awesome.'
    tokenized_string = tokenizer_en.encode(sample_string)
    for ts in tokenized_string:
        print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

    # The encoder uses BERT tokenizer.

    # In[ ]:

    tokenizer_zh = FullTokenizer(
        vocab_file='chinese_L-12_H-768_A-12/vocab.txt', do_lower_case=True)

    test_tokens = tokenizer_zh.tokenize('今天天气不错额。')
    test_ids = tokenizer_zh.convert_tokens_to_ids(['[CLS]'] + test_tokens + ['[SEP]'])
    print('tokens:', test_tokens)
    print('ids:', test_ids)
    print('convert_ids_to_tokens:', tokenizer_zh.convert_ids_to_tokens(test_ids))


    def encode(zh, en, seq_length=MAX_SEQ_LENGTH):
        tokens_zh = tokenizer_zh.tokenize(tf.compat.as_text(zh.numpy()))
        lang1 = tokenizer_zh.convert_tokens_to_ids(['[CLS]'] + tokens_zh + ['[SEP]'])
        if len(lang1) < seq_length:
            lang1 = lang1 + list(np.zeros(seq_length - len(lang1), 'int32'))

        # insert SOS and EOS
        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
            tf.compat.as_text(en.numpy())) + [tokenizer_en.vocab_size + 1]
        if len(lang2) < seq_length:
            lang2 = lang2 + list(np.zeros(seq_length - len(lang2), 'int32'))

        return lang1, lang2



    def filter_max_length(x, y, max_length=MAX_SEQ_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)

    train_dataset = train_examples.map(
        lambda zh, en: tf.py_function(encode, [zh, en], [tf.int32, tf.int32]))
    train_dataset = train_dataset.filter(filter_max_length)

    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(20000).padded_batch(
        BATCH_SIZE, padded_shapes=([-1], [-1]), drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_examples.map(
        lambda zh, en: tf.py_function(encode, [zh, en], [tf.int32, tf.int32]))
    val_dataset = val_dataset.filter(filter_max_length)
    val_dataset = val_dataset.padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))




    return train_dataset, val_dataset, tokenizer_en, tokenizer_zh


if __name__ == '__main__':
    get_tokenizer(100, 64)