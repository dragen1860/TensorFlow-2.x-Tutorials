# tf-eager-fasterrcnn

Faster R-CNN R-101-FPN model was implemented with tensorflow eager execution. 

# Requirements

- Cuda 9.0
- Python 3.5
- TensorFlow 1.11
- cv2

# Usage

see `train_model.ipynb` and `inspect_model.ipynb`

### Download trained Faster R-CNN

- [百度网盘](https://pan.baidu.com/s/1I5PGkpvnDSduJnngoWuktQ)


# Updating

- [ ] evaluation utils
- [ ] training without for-loop when batch size > 1
- [ ] TTA (multi-scale testing and flip)
- [ ] online hard examples mining
- [ ] multi-scale training
- [ ] soft-nms
- [ ] balanced sampling and negative samples training (maybe)


# Acknowledgement

This work builds on many excellent works, which include:

- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)