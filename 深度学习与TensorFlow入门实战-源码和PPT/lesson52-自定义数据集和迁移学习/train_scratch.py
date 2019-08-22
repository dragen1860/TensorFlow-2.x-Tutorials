import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses
from    tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


from    pokemon import load_pokemon, normalize, denormalize
from    resnet import ResNet


def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [244, 244])

    x = tf.image.random_flip_left_right(x)
    # x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [224,224,3])

    # x: [0,255]=> -1~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=5)

    return x, y


batchsz = 256

# creat train db
images, labels, table = load_pokemon('pokemon',mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)
# crate validation db
images2, labels2, table = load_pokemon('pokemon',mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batchsz)
# create test db
images3, labels3, table = load_pokemon('pokemon',mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)


resnet = keras.Sequential([
    layers.Conv2D(16,5,3),
    layers.MaxPool2D(3,3),
    layers.ReLU(),
    layers.Conv2D(64,5,3),
    layers.MaxPool2D(2,2),
    layers.ReLU(),
    layers.Flatten(),
    layers.Dense(64),
    layers.ReLU(),
    layers.Dense(5)
])


# resnet = ResNet(5)
resnet.build(input_shape=(4, 224, 224, 3))
resnet.summary()

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=5
)

resnet.compile(optimizer=optimizers.Adam(lr=1e-3),
               loss=losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
resnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=100,
           callbacks=[early_stopping])
resnet.evaluate(db_test)