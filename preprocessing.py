import pandas as pd
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import numpy as np
import re
# import six.moves.cPickle as pkl
import pickle as pkl
from collections import Counter
import itertools
import math

labels = {'pos':1, 'neg':0}
trainfile = pd.DataFrame()
trainfile = pd.read_csv('./data/movie_data_train.csv')
testfile = pd.read_csv('./data/movie_data_test.csv')

def preprocessor(text):
    # print("Remove punctuations, number... lowercase..")
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('\d+','', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emoticons).replace('-', '')
    return text

def remove_stopwords(text):
    print("Remove stop words...")
    #Need to download nltk stopwords if you dont have
    # nltk.download('stopwords')

    stop = stopwords.words('english')
    result = []
    for item in text:
        item = [x for x in item if x not in stop]
        result.append(item)
    return result

def load_train_test():
    print("Load train and test set...")
    trainfile['review'] = trainfile['review'].apply(preprocessor)
    testfile['review'] = testfile['review'].apply(preprocessor)
    print("Finished preprocessing")
    x_train = trainfile.loc[:, 'review'].values
    y_train = trainfile.loc[:, 'sentiment'].values
    test = testfile.loc[:, 'review'].values

    print("Start Stemming...")
    st = LancasterStemmer()
    x_train = [[st.stem(word) for word in sentence.split(" ")] for sentence in x_train]
    test = [[st.stem(word) for word in sentence.split(" ")] for sentence in test]
    print("Finished Stemming")

    f = open('./data/stem.pkl', 'wb')
    pkl.dump((x_train, y_train, test), f, -1)
    f.close()

    return x_train, y_train, test

# too slow to do stemming, so I save the packle.
def loadpkl():
    f = open('./data/stem.pkl', 'rb')
    result = pkl.load(f)
    f.close()
    x_train = result[0]
    y_train = result[1]
    test = result[2]
    return x_train, y_train, test


def padding(x_text, maxlen = 100, top_end_ratio = 0.5):
    bookmark = math.ceil(maxlen * top_end_ratio)
    padding_word = ""
    new_x_text = []
    for i in range(len(x_text)):
        if len(x_text[i]) <= maxlen:
            num_padding = maxlen - len(x_text[i])
            padded_text = x_text[i] + [padding_word] * num_padding
            new_x_text.append(padded_text)
        else:
            new_x_text.append(x_text[i][0:int(bookmark)] + x_text[i][int(bookmark):int(maxlen)])
    x_text = new_x_text
    return x_text

def build_dict(sentences):
    print("Build dictionary...")
    wordcount = dict()
    for sentence in sentences:
        for word in sentence:
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
    counts = list(wordcount.values())
    keys = list(wordcount.keys())

    sorted_idx = np.argsort(counts)[::-1]
    worddict = dict()
    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print(np.sum(counts), ' total words ', len(keys), ' unique words')
    return worddict

# There might be words in test set that is out of our dictionary
def grab_data(sentences, dictionary):
    print("Translating...")
    seqs = [None] * len(sentences)
    for idx, sentence in enumerate(sentences):
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in sentence]
    return seqs

def only_once(counter):
    oov_list = []
    for item in counter:
        if counter[item]==1:
            oov_list.append(item)
    return oov_list

def analysis(sentences):
    word_counters = Counter(itertools.chain(*sentences))
    oov_list = only_once(word_counters)
    print("Vocabulary size is:", len(word_counters))
    print(len(oov_list),"words only appeared once!")
    len_list = [len(sentence) for sentence in sentences]
    len_max = max(len_list)
    len_mean = sum(len_list)/len(len_list)
    len_min = min(len_list)
    print("Max:",len_max,"Min",len_min,"Mean",len_mean)

if __name__== '__main__':
    x_train, y_train, test = load_train_test()
    # x_train, y_train, test = loadpkl()
    method = "NN"

    if method == "NN":
        # for neural networks
        x_train = padding(x_train, maxlen = 100, top_end_ratio = 0.5)
        test = padding(test)

    raw_train = x_train

    # print(raw_train[1:10])

    raw_test = test
    dictionary = build_dict(x_train)
    x_train = grab_data(x_train, dictionary)
    test = grab_data(test,dictionary)

    analysis(raw_train)

    f = open('./data/cleaned.pkl', 'wb')
    pkl.dump((raw_train, x_train, y_train), f, -1)
    f.close()

    f = open('./data/test.pkl', 'wb')
    pkl.dump((raw_test, test), f, -1)
    f.close()

    f = open('./data/dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()
