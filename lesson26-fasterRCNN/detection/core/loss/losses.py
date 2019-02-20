import tensorflow as tf
from    tensorflow import keras


def smooth_l1_loss(y_true, y_pred):
    '''Implements Smooth-L1 loss.
    
    Args
    ---
        y_true and y_pred are typically: [N, 4], but could be any shape.
    '''
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss



def rpn_class_loss(target_matchs, rpn_class_logits):
    '''RPN anchor classifier loss.
    
    Args
    ---
        target_matchs: [batch_size, num_anchors]. Anchor match type. 1=positive,
            -1=negative, 0=neutral anchor.
        rpn_class_logits: [batch_size, num_anchors, 2]. RPN classifier logits for FG/BG.
    '''

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = tf.cast(tf.equal(target_matchs, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(tf.not_equal(target_matchs, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=anchor_class,
    #                                               logits=rpn_class_logits)

    num_classes = rpn_class_logits.shape[-1]
    # print(rpn_class_logits.shape)
    loss = keras.losses.categorical_crossentropy(tf.one_hot(anchor_class, depth=num_classes),
                                                 rpn_class_logits, from_logits=True)

    
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
    return loss


def rpn_bbox_loss(target_deltas, target_matchs, rpn_deltas):
    '''Return the RPN bounding box loss graph.
    
    Args
    ---
        target_deltas: [batch, num_rpn_deltas, (dy, dx, log(dh), log(dw))].
            Uses 0 padding to fill in unsed bbox deltas.
        target_matchs: [batch, anchors]. Anchor match type. 1=positive,
            -1=negative, 0=neutral anchor.
        rpn_deltas: [batch, anchors, (dy, dx, log(dh), log(dw))]
    '''
    def batch_pack(x, counts, num_rows):
        '''Picks different number of values from each row
        in x depending on the values in counts.
        '''
        outputs = []
        for i in range(num_rows):
            outputs.append(x[i, :counts[i]])
        return tf.concat(outputs, axis=0)
    
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = tf.where(tf.equal(target_matchs, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_deltas = tf.gather_nd(rpn_deltas, indices)

    # Trim target bounding box deltas to the same length as rpn_deltas.
    batch_counts = tf.reduce_sum(tf.cast(tf.equal(target_matchs, 1), tf.int32), axis=1)
    target_deltas = batch_pack(target_deltas, batch_counts,
                              target_deltas.shape.as_list()[0])

    loss = smooth_l1_loss(target_deltas, rpn_deltas)
    
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
    
    return loss





def rcnn_class_loss(target_matchs_list, rcnn_class_logits_list):
    '''Loss for the classifier head of Faster RCNN.
    
    Args
    ---
        target_matchs_list: list of [num_rois]. Integer class IDs. Uses zero
            padding to fill in the array.
        rcnn_class_logits_list: list of [num_rois, num_classes]
    '''
    
    class_ids = tf.concat(target_matchs_list, 0)
    class_logits = tf.concat(rcnn_class_logits_list, 0)
    class_ids = tf.cast(class_ids, 'int64')
    
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=class_ids,
    #                                               logits=class_logits)

    num_classes = class_logits.shape[-1]
    # print(class_logits.shape)
    loss = keras.losses.categorical_crossentropy(tf.one_hot(class_ids, depth=num_classes),
                                                 class_logits, from_logits=True)


    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
    return loss


def rcnn_bbox_loss(target_deltas_list, target_matchs_list, rcnn_deltas_list):
    '''Loss for Faster R-CNN bounding box refinement.
    
    Args
    ---
        target_deltas_list: list of [num_positive_rois, (dy, dx, log(dh), log(dw))]
        target_matchs_list: list of [num_rois]. Integer class IDs.
        rcnn_deltas_list: list of [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    '''
    
    target_deltas = tf.concat(target_deltas_list, 0)
    target_class_ids = tf.concat(target_matchs_list, 0)
    rcnn_deltas = tf.concat(rcnn_deltas_list, 0)

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
    
    # Gather the deltas (predicted and true) that contribute to loss
    rcnn_deltas = tf.gather_nd(rcnn_deltas, indices)

    # Smooth-L1 Loss
    loss = smooth_l1_loss(target_deltas, rcnn_deltas)
    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)

    return loss
