# AI and ML Tutorial Repository

This repository houses tutorials created for the iTrustAI SSHRC Partnership Grant. These tutorials will grow over time and will be used in hands-on workshops for training purposes.

---

# Natural Language Processing
Natural Language Processing (NLP) is the field focusing on developing methods and tools for understanding and generating human language. As such NLP actually covers a wide range of technologies, with many different applications, that are often used in concert with one other. For instance, a type of text classification such as sentiment analysis might try to utilize part of speech (POS) tagging to disambiguiate word meaning through the POS tag, and therefor improve performance on the text classification task. In this section we will introduce code to train models for many of the core NLP tasks.

---

## Part of Speech (POS) Tagging 
|          | Category      | Descriptions | Link |
|-------------|---|------------------------------|--------------------------------------|
|1|POS Tagging| POS with spaCy|[notebook](pos_tagging/POS_Tagging_Spacy_.ipynb)|
|2|POS Tagging| Train BiLSTM with PyTorch from Scratch|[notebook](pos_tagging/Bilstm_POS_Tutorial_.ipynb)|
|3|POS Tagging| Finetune with BERT from Scratch|[notebook](pos_tagging/BERT_POS_Tagging.ipynb)|

---

## Named Entity Recognition (NER) 
|          | Category      | Descriptions | Link |
|-------------|---|------------------------------|--------------------------------------|
|1|Named Entity Recognition| Introduction to NER and out-of-box solution with Spacy|[notebook](named_entity_recognition/Introduction%20to%20NER%20and%20out-of-box%20solution%20with%20Spacy.ipynb)|
|2|Named Entity Recognition| Train BiLSTM with PyTorch from Scratch|[notebook](named_entity_recognition/NER_Train_BiLSTM_with_PyTorch_from_Scratch.ipynb)|
|3|Named Entity Recognition| Fine-tune BERT with Huggingface |[notebook](named_entity_recognition/NER_Fine_tune_BERT_with_Huggingface.ipynb)|

---

## Text Classification 
Text classification aims to assign a given text to one or more categories. We can find a wide range of real-world applications of text classification, such as spam filtering and sentiment analysis. In this section, two tutorials are included. We discuss what text classification is and solve a classification task in the first tutorial. The second tutorial address a classification task using a Transformer-based deep learning model.
 
|          | Category      | Descriptions | Link |
|-------------|---|------------------------------|--------------------------------------|
|1|Text Classification|Intro and Classical Machine Learning|[notebook](text_classification/Text_Classification1.ipynb)|
|2|Text Classification|Deep Learning (BERT) |[notebook](text_classification/Text_classification_BERT.ipynb)|

--- 

## Machine Translation (MT)
Machine Translation aims to learn a automatic system to translate a given text from a language to another language. This section includes a tutorial of neural-based machine translation. We introduce a important architecture in machine translation: [sequence to sequence network](https://arxiv.org/abs/1409.3215), in which two recurrent neural networks work together to transform one sequence (e.g., sentence) to another. 

|          | Category      | Descriptions | Link |
|-------------|---|------------------------------|--------------------------------------|
|1 |Machine Translation|Seq2seq |[notebook](machine_translation/Machine_Translation_seq2seq.ipynb)|

--- 


# Speech Processing 
Speech Processing deals with extracting information or manipulating, human speech. Several challenges are present in speech processing compared to text-based NLP, we illustrate two such issues. Audio can be represented in several different ways (e.g. raw wav files, log mel spectrograms, Mel Frequency Cepstral Coefficients etc.) all of which will represents relatively long sequence lengths compared to text, making model training difficult if simply trained on the whole sequence. Data availability is also significantly less than as compared to text, greatly impacting low-resources languages where it may be difficult to find enough labeled data to make high quality speech processing models. 

## Automatic Speech Recognition (ASR)
### What is ASR?
ASR is the task of taking speech audio files and automatically creating transcriptions for the audio.

### Example ASR Tutorials

The easiest starting point for getting an ASR model running currently is to use a pretrained model, such as wav2vec 2.0. [Fine-tuning for English Systems](https://github.com/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb) is HuggingFace's tutorial on using the pre-trained English wav2vec 2.0 model (trained via a self-supervised process) and fine-tune it as an ASR model using Connectionist Temporal Classification (CTC), a common ASR technique to align recognized sounds to transcript letters. To improve results a language model and CTCdecoding are often used  [PyCTCdecode Example](https://github.com/kensho-technologies/pyctcdecode/blob/main/tutorials/02_pipeline_huggingface.ipynb)


### Tools / Tutorials

|          | Category      | Descriptions | Link |
|-------------|---|------------------------------|--------------------------------------|
| Fine-tuning wav2vec 2.0 | Supervised Learning, Transfer Learning |  Using pre-trained Speech Models can alleviate the time and data requirement to train an ASR system, while still performing near state of the art. This tutorial uses HuggingFace's Transformers library to streamline creating these models.                          |  [[Fine-tuning Models for English ASR](https://github.com/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb)]; [[Fine-tuning Models for non-English ASR](https://github.com/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb)]     |
| ESPnet Speech toolkit | Supervised Learning |  ESPnet is an End-to-End Speech Processing Toolkit, similar to the Kaldi ASR toolkit, but using purely deep neural nets for models. It is a flexible system, however, many high power recipes provided might be very resources (GPU/RAM) and time demanding. Supports ASR, Speech Translation, Text-to-Speech and more                         |   [Github](https://github.com/espnet/espnet) and [ASR example](https://colab.research.google.com/github/espnet/notebook/blob/master/espnet2_asr_realtime_demo.ipynb) [Note: ESPnet2 is under development]   |

---

# Image Processing 
Forthcoming!
- Object Recognition
- Image Caption Generation
- ... ... ...
- 
---

# Practical Machine Learning 
Forthcoming!
- Working with data for ML
- How to train a neural network
- ... ... ...


