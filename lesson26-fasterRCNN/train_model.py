import  os
import  tensorflow as tf
from    tensorflow import keras
import  numpy as np
from    matplotlib import pyplot as plt
import  visualize

from    detection.datasets import coco, data_generator
from    detection.datasets.utils import get_original_image
from    detection.models.detectors import faster_rcnn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)
assert tf.__version__.startswith('2.')
tf.random.set_seed(22)
np.random.seed(22)

img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)
batch_size = 1
train_dataset = coco.CocoDataSet('/scratch/llong/datasets/coco2017/', 'train',
                                 flip_ratio=0.5,
                                 pad_mode='fixed',
                                 mean=img_mean,
                                 std=img_std,
                                 scale=(800, 1216))
num_classes = len(train_dataset.get_categories())
train_generator = data_generator.DataGenerator(train_dataset)
train_tf_dataset = tf.data.Dataset.from_generator(
    train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.batch(batch_size).prefetch(100).shuffle(100)
# train_tf_dataset = train_tf_dataset.padded_batch(
#     batch_size, padded_shapes=([None, None, None], [None], [None, None], [None]))

model = faster_rcnn.FasterRCNN(num_classes=num_classes)
optimizer = keras.optimizers.SGD(1e-3, momentum=0.9, nesterov=True)

img, img_meta, bboxes, labels = train_dataset[6] # [N, 4], shape:[N]=data:[62]
rgb_img = np.round(img + img_mean)
ori_img = get_original_image(img, img_meta, img_mean)
# visualize.display_instances(rgb_img, bboxes, labels, train_dataset.get_categories())


batch_imgs = tf.convert_to_tensor(np.expand_dims(img, 0)) # [1, 1216, 1216, 3]
batch_metas = tf.convert_to_tensor(np.expand_dims(img_meta, 0)) # [1, 11]

# dummpy forward to build network variables
_ = model((batch_imgs, batch_metas), training=False)

proposals = model.simple_test_rpn(img, img_meta)
res = model.simple_test_bboxes(img, img_meta, proposals)
visualize.display_instances(ori_img, res['rois'], res['class_ids'],
                            train_dataset.get_categories(), scores=res['scores'])
plt.savefig('image_demo_random.png')

model.load_weights('weights/faster_rcnn.h5', by_name=True)

proposals = model.simple_test_rpn(img, img_meta)
res = model.simple_test_bboxes(img, img_meta, proposals)
visualize.display_instances(ori_img, res['rois'], res['class_ids'],
                            train_dataset.get_categories(), scores=res['scores'])
plt.savefig('image_demo_ckpt.png')

for epoch in range(100):

    loss_history = []
    for (batch, inputs) in enumerate(train_tf_dataset):

        batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = model(
                (batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)

            loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_history.append(loss_value.numpy())

        if batch % 10 == 0:
            print('epoch', epoch, batch, np.mean(loss_history))

            # img, img_meta = batch_imgs[0].numpy(), batch_metas[0].numpy()
            # ori_img = get_original_image(img, img_meta, img_mean)
            # proposals = model.simple_test_rpn(img, img_meta)
            # res = model.simple_test_bboxes(img, img_meta, proposals)
            # visualize.display_instances(ori_img, res['rois'], res['class_ids'],
            #                             train_dataset.get_categories(), scores=res['scores'])
            # plt.savefig('images/%d-%d.png' % (epoch, batch))
