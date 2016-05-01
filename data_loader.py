import math
import numpy as np
import six.moves.cPickle as pkl

def load_data():
    f = open('./data/cleaned.pkl','rb')
    train = pkl.load(f)
    f.close()
    raw_train = train[0]
    x_train = train[1]
    y_train = train[2]

    f = open('./data/test.pkl','rb')
    test = pkl.load(f)
    f.close()
    raw_test = test[0]
    x_test = test[1]

    f = open('./data/dict.pkl','rb')
    dictionary = pkl.load(f)
    f.close()
    y_train = convert_y(y_train)
    y_train = convert_y(y_train)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(y_train)
    return raw_train, raw_test, x_train, y_train, x_test, dictionary

def convert_y(y_train):
    y_labels = []
    for i in y_train:
        if i:
            y_labels.append([0,1])
        else:
            y_labels.append([1,0])
    return y_labels

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    print("before is:",type(data))
    data = np.array(data)
    print("type of data is:", type(data))
    # print(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
