# Implementation of Vision Language Model From Scratch

_(Work in Progress...)_

This repository contains the implementation of a Vision-Language Model (PaliGemma) built from scratch as a personal project.

![PaliGemma Architecture.](media/architecture.png)

# Architecture

This VLM is composed of three main components:

- SigLIP, an image encoder which was constrastively pretrained at large scale with sigmoid loss. Achieved SoTA performance, especially for its small size (checkpoints available).

- Gemma-2B, a decoder-only language model which has a great balance between performance and size (checkpoints available).

- A linear layer that projects the SigLIP's output tokens to the same dimensions as Gemma 2B's vocab tokens, such that they can be concatenated.

The image is fed into the SigLIP encoder, which outputs a sequence of N<sub>img</sub> tokens. The text is converted into N<sub>txt</sub> tokens using the Gemma's Sentence Piece tokenizer and embedded with Gemma's vocabulary embedding layer. The image tokens are then projected with the linear layer. Then the sequence of image tokens and text tokens are concatenated and fed into the Gemma-2B decoder as follows:

<div align="center">
  <img src="media/prefix-lm-masking.png" alt="PaliGemma Architecture" width="500" />
</div>

<br>

In this implementation, the images are always resized to 224x224 pixels, corresponding to 256 tokens which are always placed in the front. The BOS token then marks the start of text tokens and a `\u` is used as a separator token. But this separator is tokenized separatly to avoid it bering merged with with the end of the prefix or the beginning of the suffix. This model uses a full unmasked attention on the input (image + prefix) and the vanilla auto-regressive mask for the output (suffix).

# Key Architectural Insights

- The model leverages KV-cache. During inference, since we cache the keys and values of the previous tokens, we only process a single token at a time, so there is nothing to mask out. Obviously, during training, you still have a causal mask.
- Surprisingly, the model uses a full unmasked attention on the images tokens AND the prefix tokens !!! And an auto-regressive mask only for the output (suffix).
- RMS Normalization 
- Group Query Attention
- Rotary Positional Embeddings
- Top-p sampling