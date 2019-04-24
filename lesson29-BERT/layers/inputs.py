from tensorflow import keras


def get_inputs(seq_len):
    """Get input layers.

    See: https://arxiv.org/pdf/1810.04805.pdf

    :param seq_len: Length of the sequence or None.
    """
    names = ['Token', 'Segment', 'Masked']
    return [keras.layers.Input(
        shape=(seq_len,),
        name='Input-%s' % name,
    ) for name in names]
