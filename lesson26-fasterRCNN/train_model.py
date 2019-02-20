import os
import tensorflow as tf
from    tensorflow import keras
import numpy as np
import visualize



from detection.datasets import coco, data_generator

from detection.datasets.utils import get_original_image

from detection.models.detectors import faster_rcnn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'




img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)




train_dataset = coco.CocoDataSet('/scratch/llong/datasets/coco2017/', 'train',
                                 flip_ratio=0.5,
                                 pad_mode='fixed',
                                 mean=img_mean,
                                 std=img_std,
                                 scale=(800, 1216))

train_generator = data_generator.DataGenerator(train_dataset)


img, img_meta, bboxes, labels = train_dataset[6]

rgb_img = np.round(img + img_mean)
ori_img = get_original_image(img, img_meta, img_mean)

visualize.display_instances(rgb_img, bboxes, labels, train_dataset.get_categories())




model = faster_rcnn.FasterRCNN(num_classes=len(train_dataset.get_categories()))


# In[7]:


batch_imgs = tf.Variable(np.expand_dims(img, 0))
batch_metas = tf.Variable(np.expand_dims(img_meta, 0))

_ = model((batch_imgs, batch_metas), training=False)


# In[8]:


model.load_weights('weights/faster_rcnn.h5', by_name=True)


# In[9]:

proposals = model.simple_test_rpn(img, img_meta)
res = model.simple_test_bboxes(img, img_meta, proposals)


# In[10]:


visualize.display_instances(ori_img, res['rois'], res['class_ids'], 
                            train_dataset.get_categories(), scores=res['scores'])


# #### use tf.data

# In[11]:


batch_size = 1

train_tf_dataset = tf.data.Dataset.from_generator(
    train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.padded_batch(
    batch_size, padded_shapes=([None, None, None], [None], [None, None], [None]))


# #### overfit a sample

# In[12]:


# optimizer = tf.train.MomentumOptimizer(1e-3, 0.9, use_nesterov=True)
optimizer = keras.optimizers.SGD(1e-3, momentum=0.9, nesterov=True)

epochs = 100

for epoch in range(epochs):
    iterator = train_tf_dataset.make_one_shot_iterator()

    loss_history = []
    for (batch, inputs) in enumerate(iterator):
    
        batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)

            loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_history.append(loss_value.numpy())

        if batch%100==0:
            print('epoch', epoch, batch, np.mean(loss_history))

    proposals = model.simple_test_rpn(img, img_meta)
    res = model.simple_test_bboxes(img, img_meta, proposals)
    visualize.display_instances(ori_img, res['rois'], res['class_ids'],
                                train_dataset.get_categories(), scores=res['scores'])





