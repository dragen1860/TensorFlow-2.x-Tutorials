import tensorflow as tf
from detection.core.bbox import geometry, transforms
from detection.utils.misc import trim_zeros



class AnchorTarget:
    """
    for every generated anchors boxes: [326393, 4],
    create its rpn_target_matchs and rpn_target_matchs
    which is used to train RPN network.
    """
    def __init__(self,
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3):
        '''
        Compute regression and classification targets for anchors.
        
        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RPN.
            target_stds: [4]. Bounding box refinement standard deviation for RPN.
            num_rpn_deltas: int. Maximal number of Anchors per image to feed to rpn heads.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rpn_deltas = num_rpn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr

    def build_targets(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        '''
        Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.

        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image 
                coordinates. batch_size = 1 usually
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.

        Returns
        ---
            rpn_target_matchs: [batch_size, num_anchors] matches between anchors and GT boxes.
                1 = positive anchor, -1 = negative anchor, 0 = neutral anchor
            rpn_target_deltas: [batch_size, num_rpn_deltas, (dy, dx, log(dh), log(dw))] 
                Anchor bbox deltas.
        '''
        rpn_target_matchs = []
        rpn_target_deltas = []
        
        num_imgs = gt_class_ids.shape[0] # namely, batchsz , 1
        for i in range(num_imgs):
            target_match, target_delta = self._build_single_target(
                anchors, valid_flags[i], gt_boxes[i], gt_class_ids[i])
            rpn_target_matchs.append(target_match)
            rpn_target_deltas.append(target_delta)
        
        rpn_target_matchs = tf.stack(rpn_target_matchs)
        rpn_target_deltas = tf.stack(rpn_target_deltas)
        
        rpn_target_matchs = tf.stop_gradient(rpn_target_matchs)
        rpn_target_deltas = tf.stop_gradient(rpn_target_deltas)
        
        return rpn_target_matchs, rpn_target_deltas

    def _build_single_target(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        '''Compute targets per instance.
        
        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)]
            valid_flags: [num_anchors]
            gt_class_ids: [num_gt_boxes]
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
        
        Returns
        ---
            target_matchs: [num_anchors]
            target_deltas: [num_rpn_deltas, (dy, dx, log(dh), log(dw))] 
        '''
        gt_boxes, _ = trim_zeros(gt_boxes) # remove padded zero boxes, [new_N, 4]
        
        target_matchs = tf.zeros(anchors.shape[0], dtype=tf.int32) # [326393]
        
        # Compute overlaps [num_anchors, num_gt_boxes] 326393 vs 10 => [326393, 10]
        overlaps = geometry.compute_overlaps(anchors, gt_boxes)

        # Match anchors to GT Boxes
        # If an anchor overlaps ANY GT box with IoU >= 0.7 then it's positive.
        # If an anchor overlaps ALL GT box with IoU < 0.3 then it's negative.
        # Neutral anchors are those that don't match the conditions above,
        # and they don't influence the loss function.
        # However, don't keep any GT box unmatched (rare, but happens). Instead,
        # match it to the closest anchor (even if its max IoU is < 0.3).
        
        neg_values = tf.constant([0, -1])
        pos_values = tf.constant([0, 1])
        
        # 1. Set negative anchors first. They get overwritten below if a GT box is
        # matched to them. [N_anchors, N_gt_boxes]
        anchor_iou_argmax = tf.argmax(overlaps, axis=1) # [326396] get clost gt boxes for each anchors
        anchor_iou_max = tf.reduce_max(overlaps, axis=[1]) # [326396] get closet gt boxes's overlap scores
        # if an anchor box overlap all GT box with IoU < 0.3, marked as -1 background
        target_matchs = tf.where(anchor_iou_max < self.neg_iou_thr, 
                                 -tf.ones(anchors.shape[0], dtype=tf.int32), target_matchs)

        # filter invalid anchors
        target_matchs = tf.where(tf.equal(valid_flags, 1),
                                 target_matchs, tf.zeros(anchors.shape[0], dtype=tf.int32))
        # if an anchor overlap with any GT box with IoU > 0.7, marked as foreground
        # 2. Set anchors with high overlap as positive.
        target_matchs = tf.where(anchor_iou_max >= self.pos_iou_thr, 
                                 tf.ones(anchors.shape[0], dtype=tf.int32), target_matchs)

        # 3. Set an anchor for each GT box (regardless of IoU value).        
        gt_iou_argmax = tf.argmax(overlaps, axis=0) # [N_gt_boxes]
        target_matchs = tf.compat.v1.scatter_update(tf.Variable(target_matchs), gt_iou_argmax, 1)
        # update corresponding value=>1 for GT boxes' closest boxes
        
        # Subsample to balance positive and negative anchors
        # Don't let positives be more than half the anchors
        ids = tf.where(tf.equal(target_matchs, 1))  # [N_pos_anchors, 1], [15, 1]
        ids = tf.squeeze(ids, 1) # [15]
        extra = ids.shape.as_list()[0] - int(self.num_rpn_deltas * self.positive_fraction) # 256*0.5
        if extra > 0: # extra means the redundant pos_anchors
            # Reset the extra random ones to neutral
            ids = tf.random.shuffle(ids)[:extra]
            target_matchs = tf.compat.v1.scatter_update(target_matchs, ids, 0)
        # Same for negative proposals
        ids = tf.where(tf.equal(target_matchs, -1)) # [213748, 1]
        ids = tf.squeeze(ids, 1)
        extra = ids.shape.as_list()[0] - (self.num_rpn_deltas - # 213748 - (256 - num_of_pos_anchors:15)
            tf.reduce_sum(tf.cast(tf.equal(target_matchs, 1), tf.int32)))
        if extra > 0: # 213507, so many negative anchors!
            # Rest the extra ones to neutral
            ids = tf.random.shuffle(ids)[:extra]
            target_matchs = tf.compat.v1.scatter_update(target_matchs, ids, 0)
        # since we only need 256 anchors, and it had better contains half positive anchors, and harlf neg .
        
        # For positive anchors, compute shift and scale needed to transform them
        # to match the corresponding GT boxes.
        ids = tf.where(tf.equal(target_matchs, 1)) # [15]
        
        a = tf.gather_nd(anchors, ids) # [369303, 4], [15] => [15, 4]
        anchor_idx = tf.gather_nd(anchor_iou_argmax, ids) # closed gt boxes index for 369303 anchors
        gt = tf.gather(gt_boxes, anchor_idx) # get closed gt boxes coordinates for ids=15
        # a: [15, 4], postive anchors, gt: [15, 4] closed gt boxes for each anchors=15
        target_deltas = transforms.bbox2delta(
            a, gt, self.target_means, self.target_stds)
        # target_deltas: [15, (dy,dx,logw,logh)]?
        padding = tf.maximum(self.num_rpn_deltas - tf.shape(target_deltas)[0], 0) # 256-15
        target_deltas = tf.pad(target_deltas, [(0, padding), (0, 0)]) #padding to [256,4], last padding 0

        return target_matchs, target_deltas