# AI IMG Classifier

To find open weight models, try these sources:

- [TIMM Benchmark list](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet-a-clean.csv)

- [Papers With Code](https://paperswithcode.com/sota/image-classification-on-imagenet)

- [Open Mixup](https://github.com/Westlake-AI/openmixup)

- [Torch Model Zoo](https://docs.pytorch.org/vision/main/models.html)


# Models

### tinynet_152

- Epoch in 30-45min
- loss 2.6, MAE 0.48, no change in 1 epoch (??)
- AVP 0.52

### tinynet_106

- Epoch in <30min
- loss 2.6, MAE 0.51, no change in 1 epoch (??)
- AVP 0.51

### tinyvit_s_224

- epoch in 1h
- loss 0.7, avg precision 0.57??

### Edgenext

- similar as above, but much worse test set scores


### Models that are not competitive

Resnet50 (obviously, it's 10 years old)

Regnet / resnext / mobilenetv2 / Swinv2

