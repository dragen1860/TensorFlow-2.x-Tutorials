import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras


# In[1]:


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


# In[2]:


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
# [b, 28, 28] => [b, 28, 28, 1]
x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(256)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

# In[3]:


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[4]:


class ConvBNRelu(keras.Model):
    
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(ch, kernelsz, strides=strides, padding=padding),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])
        
        
    def call(self, x, training=None):
        
        x = self.model(x, training=training)
        
        return x 
    
        
        


# In[5]:


class InceptionBlk(keras.Model):
    
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        
        self.ch = ch
        self.strides = strides
        
        self.conv1 = ConvBNRelu(ch, strides=strides)
        self.conv2 = ConvBNRelu(ch, kernelsz=3, strides=strides)
        self.conv3_1 = ConvBNRelu(ch, kernelsz=3, strides=strides)
        self.conv3_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
        
        self.pool = keras.layers.MaxPooling2D(3, strides=1, padding='same')
        self.pool_conv = ConvBNRelu(ch, strides=strides)
        
        
    def call(self, x, training=None):
        
        
        x1 = self.conv1(x, training=training)

        x2 = self.conv2(x, training=training)
                
        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)
                
        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=training)
        
        # concat along axis=channel
        x = tf.concat([x1, x2, x3_2, x4], axis=3)
        
        return x


# In[6]:


class Inception(keras.Model):
    
    def __init__(self, num_layers, num_classes, init_ch=16, **kwargs):
        super(Inception, self).__init__(**kwargs)
        
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_layers = num_layers
        self.init_ch = init_ch
        
        self.conv1 = ConvBNRelu(init_ch)
        
        self.blocks = keras.models.Sequential(name='dynamic-blocks')
        
        for block_id in range(num_layers):
            
            for layer_id in range(2):
                
                if layer_id == 0:
                    
                    block = InceptionBlk(self.out_channels, strides=2)
                    
                else:
                    block = InceptionBlk(self.out_channels, strides=1)
                    
                self.blocks.add(block)
            
            # enlarger out_channels per block    
            self.out_channels *= 2
            
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)
        
        
    def call(self, x, training=None):
        
        out = self.conv1(x, training=training)
        
        out = self.blocks(out, training=training)
        
        out = self.avg_pool(out)
        out = self.fc(out)
        
        return out    
            
        


# In[7]:


# build model and optimizer
batch_size = 32
epochs = 100
model = Inception(2, 10)
# derive input shape for every layers.
model.build(input_shape=(None, 28, 28, 1))
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
criteon = keras.losses.CategoricalCrossentropy(from_logits=True)

acc_meter = keras.metrics.Accuracy()


for epoch in range(100):

    for step, (x, y) in enumerate(db_train):

        with tf.GradientTape() as tape:
            # print(x.shape, y.shape)
            # [b, 10]
            logits = model(x)
            # [b] vs [b, 10]
            loss = criteon(tf.one_hot(y, depth=10), logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 10 == 0:
            print(epoch, step, 'loss:', loss.numpy())


    acc_meter.reset_states()
    for x, y in db_test:
        # [b, 10]
        logits = model(x, training=False)
        # [b, 10] => [b]
        pred = tf.argmax(logits, axis=1)
        # [b] vs [b, 10]
        acc_meter.update_state(y, pred)

    print(epoch, 'evaluation acc:', acc_meter.result().numpy())
