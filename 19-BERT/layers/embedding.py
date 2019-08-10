from tensorflow import keras
import tensorflow.keras.backend as K
from keras_pos_embd import PositionEmbedding
from keras_layer_normalization import LayerNormalization


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return [super(TokenEmbedding, self).compute_output_shape(input_shape), (self.input_dim, self.output_dim)]

    def compute_mask(self, inputs, mask=None):
        return [super(TokenEmbedding, self).compute_mask(inputs, mask), None]

    def call(self, inputs):
        return [super(TokenEmbedding, self).call(inputs), self.embeddings]


def get_embedding(inputs, token_num, pos_num, embed_dim, dropout_rate=0.1, trainable=True):
    """Get embedding layer.

    See: https://arxiv.org/pdf/1810.04805.pdf

    :param inputs: Input layers.
    :param token_num: Number of tokens.
    :param pos_num: Maximum position.
    :param embed_dim: The dimension of all embedding layers.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: The merged embedding layer and weights of token embedding.
    """
    embeddings = [
        TokenEmbedding(
            input_dim=token_num,
            output_dim=embed_dim,
            mask_zero=True,
            trainable=trainable,
            name='Embedding-Token',
        )(inputs[0]),
        keras.layers.Embedding(
            input_dim=2,
            output_dim=embed_dim,
            trainable=trainable,
            name='Embedding-Segment',
        )(inputs[1]),
    ]
    embeddings[0], embed_weights = embeddings[0]
    embed_layer = keras.layers.Add(name='Embedding-Token-Segment')(embeddings)
    embed_layer = PositionEmbedding(
        input_dim=pos_num,
        output_dim=embed_dim,
        mode=PositionEmbedding.MODE_ADD,
        trainable=trainable,
        name='Embedding-Position',
    )(embed_layer)
    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='Embedding-Dropout',
        )(embed_layer)
    else:
        dropout_layer = embed_layer
    norm_layer = LayerNormalization(
        trainable=trainable,
        name='Embedding-Norm',
    )(dropout_layer)
    return norm_layer, embed_weights


class EmbeddingSimilarity(keras.layers.Layer):
    """Calculate similarity between features and token embeddings with bias term."""

    def __init__(self,
                 initializer='zeros',
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        """Initialize the layer.

        :param output_dim: Same as embedding output dimension.
        :param initializer: Initializer for bias.
        :param regularizer: Regularizer for bias.
        :param constraint: Constraint for bias.
        :param kwargs: Arguments for parent class.
        """
        super(EmbeddingSimilarity, self).__init__(**kwargs)
        self.supports_masking = True
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)
        self.bias = None

    def get_config(self):
        config = {
            'initializer': keras.initializers.serialize(self.initializer),
            'regularizer': keras.regularizers.serialize(self.regularizer),
            'constraint': keras.constraints.serialize(self.constraint),
        }
        base_config = super(EmbeddingSimilarity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.bias = self.add_weight(
            shape=(input_shape[1][0],),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            name='bias',
        )
        super(EmbeddingSimilarity, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + (input_shape[1][0],)

    def compute_mask(self, inputs, mask=None):
        return mask[0]

    def call(self, inputs, mask=None, **kwargs):
        inputs, embeddings = inputs
        outputs = K.bias_add(K.dot(inputs, K.transpose(embeddings)), self.bias)
        return keras.activations.softmax(outputs)
