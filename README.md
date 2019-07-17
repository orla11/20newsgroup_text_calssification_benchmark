# 20 Newsgroup Text Calssification Benchmark

NOTE: this is an academic project made for the course in Language Processing Technology.

The assignment reguards the realisation of a benchmark which includes some of the commonly used NLP techniques for text classification. More details below.

Some notebooks where I discuss and develop different models in order to classify the 20newsgroup dataset and prepare a resulting benchmark document with all the results obtained.

## Requirements
  pip install -r requirements.txt

## Data preparation
  - data_preprocessing.ipynb
  
## Models
  (class2sequence_noHidden.ipynb)
  - LSTM + MLP
  - LSTM + MLP (pretrained glove embeddings)
  
  (class2sequence.ipynb)
  - LSTM + Dense Output Layer
  - LSTM + Dense Output Layer (pretrained glove embeddings)
  
  - Bag of Words + MLP (bow.ipynb)
  - Bag of Embeddings + MLP (SOWE: sum of word embeddings) (sowe.ipynb)
  
## Other
  - /results: a bunch of pdfs with the results obtained
  - /pkls: pkls created for the sake of semplicity in order to speed up the process of reproducing the classification
  - /images: image folder
  - /graps: where all the resulting graphs for each model are stored
