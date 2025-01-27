# Implementation of Vision Language Model From Scratch

_(Work in Progress...)_

This repository contains the implementation of a Vision-Language Model (VLM) built from scratch as a personal project. This model is inspired from the PaliGemma Model.

![PaliGemma Architecture.](images/architecture.png)

# Architecture

The model is composed of three main components:

- SigLIP, an image encoder which was constrastively pretrained at large scale with sigmoid loss. Achieved SoTA performance, especially for its small size (checkpoints available).

- Gemma-2B, a decoder-only language model which has a great balance between performance and size (checkpoints available).

- A linear layer that projects the SigLIP's output tokens to the same dimensions as Gemma 2B's vocab tokens, such that they can be concatenated.

The image is fed into the SigLIP encoder, which outputs a sequence of N<sub>img</sub> tokens. The text is converted into N<sub>txt</sub> tokens using the Gemma's Sentence Piece tokenizer and embedded with Gemma's vocabulary embedding layer. The image tokens are then projected with the linear layer. Then the sequence of image tokens and text tokens are concatenated and fed into the Gemma-2B decoder as follows:

<div align="center">
  <img src="images/prefix-lm-masking.png" alt="PaliGemma Architecture" width="500" />
</div>

<br>

In our implementation, the images are always resized to 224x224 pixels, corresponding to 256 tokens which are always placed in the front. The BOS token then marks the start of text tokens and a `\u` is used as a separator token. But this separator is tokenized separatly to avoid it bering merged with with the end of the prefix or the beginning of the suffix. This model uses a full unmasked attention on the input (image + prefix) and the vanilla auto-regressive mask for the output (suffix).

# Training

PaliGemma's pretraining is limited to 'text' covering natural language, object detection and instant segmentation. 