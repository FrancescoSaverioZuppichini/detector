# [WIP 🚧] Object Detection Model

*This is a WIP project, this README is its bible and it will change over time*

## Goal & Motivation

**The goal of this repo is not to create the best developer friendly object detectiol model**

Instead of focusing on a family of different sizes models, I aim to train one middle size one (between 80M - 100M params) and provide (hopefully) quantized/smaller models using different techniques using that one model.

These are the main key points:

- Incorporates the latest techniques and algorithms from research and the community.
- Provides state-of-the-art performance on various object detection tasks.
- Designed to be fast and efficient, with a focus on ease-of-use and flexibility.
- Easy export to ONNX for deployment on a variety of platforms.
- Fast data loading for efficient training and evaluation.

The main goal is that it must be easy to use and deploy, no research spaghetti code.


### SOTA means nothing

SOTA on common datasets (COCO) means nothing. Most of the datasets have wrong labels, what I will focus on is to ensure the models has competitive performance on the most used research datasets and real-life ones. 

**The model has to be fast and easy to finetune**

## Current Limitations

Most of the current models have one or more of the following issue

- slow data loading
- spaghetti code
- custom post processing (nms) not done in exportable to onnx pytorch code
- weak baseline models (backbones pretrain only on IN/COCO etc)
- anchor boxes
- bad packaging
- bad doc
- they are made by siths

## Plan of attack

The plan of attack is the follow

- Get something ready to train asap while keeping the design extendable
- Focus on the data pipeline, memmap dataset, augmentation on batches on GPU
- Quick but meaningfull hyperparams exploration
- Train on something big, e.g. Object365 or the Florence dataset (it's not out there, I've written to them but no reply)
- Everything (even the preprocessing part) must be exportable to onnx
- Provide developer friendly solution, such as docker files

### Papers & Resources

**[Exploring Plain Vision Transformer Backbones for Object Detection](https://arxiv.org/abs/2203.16527)**<br>
This papers implements a very simple fpn showin how you don't need hierchical features with ViTs. It also introduces a couple of tricks to work with big images, such as [`window partition`](https://github.com/facebookresearch/detectron2/blob/d779ea63faa54fe42b9b4c280365eaafccb280d6/detectron2/modeling/backbone/vit.py#L216)

**[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)**<br>
It shows how you can remove positional embeddings by using a depth wise 1x1 conv with zero padding in the MLP

**[What Makes for End-to-End Object Detection?](https://arxiv.org/abs/2012.05780)**<br>
This paper propose a new loss that incorporate predicted location cost into the bboxes assignment, removing the need of `nms`.

**[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)**<br>
Clip paper, interesting enough, everybody in the research uses f\*cking s\*itty backbones. E.g. pretrained on IN. We will use the Clip's ViT

**[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** <br>
We will freeze the backbone and train only the neck. We will test out if for finetuning, adjusting the neck's weights with Lora is enough and/or also adjusting the backbone weights with Lora. The catch is that we will never train the backbone ever again fully.

**[Tensordict](https://github.com/pytorch-labs/tensordict)** <br>
This library makes it easier to create a memmap dict-like tensor that can be efficiently processed in batches

### Contributing
We welcome contributions from the community! If you have an idea for a new feature or improvement, please open an issue or submit a pull request.
