import os
import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedShuffleSplit

from nltk import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from keras.utils import to_categorical

import pickle
import mmap
import re
import string
import tqdm

REPLACE_BY_SPACE_RE     = re.compile('[/(){}\[\]\|@,;]')
REPLACE_ESCAPE_SEQS     = re.compile('[\n\t]+')
BAD_SYMBOLS_RE          = re.compile('[^0-9a-z #+_\']')
DASH                    = re.compile('-\s')
POINTS_REMOVAL_RE       = re.compile('[.]+')

GLOVE_PATH              = './glove.6B'
GLOVE_FILENAME          = 'glove.6B.300d.txt'
PKLS_PATH               = './pkls'

EMBEDDING_DIM           = 300
NUM_CALSSES             = 20

def get_categories():
    categories = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
    return categories.target_names

def tokenize(text,clean=False,stemming=False,lemmatize=False):
    '''
    Rimozione punteggiatura
    Seleziono parole che hanno lunghezza compresa tra 2 e 15 caratteri
    '''

    if clean:
        text = clean_text(text,stemming,lemmatize)

    tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return  [f.lower() for f in tokens if f and f.lower() not in stopwords.words('english')]

def get20News():
    x_data, labels, labelToName = [], [], {}
    dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)

    for i, document in enumerate(tqdm.tqdm(dataset['data'])):
        
        stopped = tokenize(document)

        if (len(stopped) == 0):
            continue

        groupIndex = dataset['target'][i]

        x_data.append(stopped)
        labels.append(groupIndex)

        labelToName[groupIndex] = dataset['target_names'][groupIndex]

    return x_data, np.array(labels), labelToName

def train_test_split(encoded_data,labels,test_size=0.2):
    # Test & Train Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1).split(encoded_data, labels)
    train_indices, test_indices = next(sss)

    x_train, x_test = encoded_data[train_indices], encoded_data[test_indices]
    y_train = to_categorical(labels[train_indices], NUM_CALSSES)
    y_test = to_categorical(labels[test_indices], NUM_CALSSES)

    return x_train, y_train, x_test, y_test, test_indices

def getEmbeddingMatrix(word_index):
    
    if not os.path.exists('./pkls/embedding_matrix.pkl'):

        embeddings_index = {}

        print('Creazione matrice embeddings...')
        with open(os.path.join(GLOVE_PATH,GLOVE_FILENAME)) as f:

            for line in tqdm.tqdm(f, total=get_num_lines(os.path.join(GLOVE_PATH,GLOVE_FILENAME))):

                values = line.split()
                word   = values[0]
                coefs  = np.asarray(values[1:],dtype='float32')
                embeddings_index[word] = coefs

        f.close()

        # Possiamo utilizzare il dizionario 'embedding_index' 
        # e il 'word_index' per calcolare la matrice di embedding
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

        for word, i in tqdm.tqdm(word_index.items()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # le parole non trovate in embedding_index saranno impostate tutte a zero
                embedding_matrix[i] = embedding_vector
                
        print('Salvo file embedding_matrix.pkl ...')
        savePkl('embedding_matrix',embedding_matrix) # salvo il dizionario per futuri lanci dello script

        return embedding_matrix
    else:
        return loadPkl('embedding_matrix')

def clean_text(text,stemming=False,lemmatize=False):

    """
        text: a string
        
        return: modified initial string
    """
    ## Remove puncuation
    #text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()

    text = " ".join(text)

    text = text.replace('\n','')
    text = text.replace('\t','')

    text = DASH.sub('', text)
    text = REPLACE_BY_SPACE_RE.sub('', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = POINTS_REMOVAL_RE.sub('', text) # replace POINTS_REMOVAL_RE symbol by space in text. substitute the matched string in POINTS_REMOVAL_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 

    ## Stemming
    if stemming:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    if lemmatize:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    return text

def get_num_lines(file_path):
    print('Counting number of lines in file...')
    fp = open(file_path,'r+')
    buf = mmap.mmap(fp.fileno(),0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def savePkl(name,dict):
    f = open(os.path.join(PKLS_PATH, name + ".pkl"),"wb")
    pickle.dump(dict,f)
    f.close()

def loadPkl(name):
    with open(os.path.join(PKLS_PATH, name + ".pkl"), 'rb') as f:
        data = pickle.load(f)
    return data