# AI and ML Tutorial Repository

# Outline
## Option A (Broad Categoy / Task)

Example:

---

# General Instructions

For each task, we want to do the following in a notebook.

- what is it?
- a small dataset
- an example tutorial, possibly just using an off-the-shelf tool (classical ML)
- a deep learning example
- list of external tools, possibly with some brief explanation about each tool (a couple of sentences, what it does, why good/bad, what languages it covers)

# Natural Language Processing
"Natural Language Processing (NLP) is an area of  research and application that explores how computers can be used  to understand and manipulate natural language text or speech  to do useful things." (Chowdhury 2003) As such NLP actually covers a wide range of techniques, with many different applications, that are often used in concert with each other. For instance, a type of Text Classification called Sentiment Analysis might try to utilize Part of Speech (POS) Tagging to disambiguiate word meaning through the POS tag, and therefor improve performance on the Text Classification task. In this section we will introduce many of the core NLP tasks, their intended purpose, and resources to aid their use, we opt to leave Speech Processing as a separate category for simplicity.



For POS and NER:

- explain what each is
- provide a sample dataset (small, in English)
- show how to train a model with deep learning (one with BiLSTM using PyTorch, and another with fine-tuning BERT use HuggingFace)
- list existing tools and explain how to use Spacy

## POS Tagging (Ife)


## NER (Weirui)


## Text Classification (Chiyu)

- What is text classification?


|          | Category      | Descriptions | Link |
|-------------|---|------------------------------|--------------------------------------|
|xx|xx|xx|xx|

## Machine Translation

--- 


# Speech Processing (Peter)
- 


## Automatic Speech Recognition (ASR)
### What is ASR
ASR is the task of taking speech audio files and automatically creating transcriptions for the audio.

### Example ASR Tutorial



### Tools / Tutorials

|          | Category      | Descriptions | Link |
|-------------|---|------------------------------|--------------------------------------|
| Fine-tuning wav2vec 2.0 | Supervised Learning, Transfer Learning |  Using pre-trained Speech Models can alleviate the time and data requirement to train an ASR system, while still performing near state of the art. This tutorial uses HuggingFace's Transformers library to streamline creating these models.                          |  [Fine-tuning for English Systems](https://github.com/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb) [Fine-tuning for non-English Systems](https://github.com/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb)     |
| ESPnet Speech toolkit | Supervised Learning |  ESPnet is an End-to-End Speech Processing Toolkit, similar to the Kaldi ASR toolkit, but using purely deep neural nets for models. It is a flexible system, however, many high power recipes provided might be very resources (GPU/RAM) and time demanding. Supports ASR, Speech Translation, Text-to-Speech and more                         |   [Github](https://github.com/espnet/espnet) and [ASR example](https://colab.research.google.com/github/espnet/notebook/blob/master/espnet2_asr_realtime_demo.ipynb) [Note: ESPnet2 underdevelopment]   |
|  |                           |      | | 
|  |                           |      | |
|   |       |                           |  | |
|   |  |   | |


# Image Processing (Chiyu)

- object recognition
- caption generation





### Citations


Chowdhury, G. G. (2003). Natural language processing. Annual review of information science and technology, 37(1), 51-89.


