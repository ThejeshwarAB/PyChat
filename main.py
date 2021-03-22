from os import path
import os.path
from tensorflow.python.framework import ops
import pickle
import json
import random
import tensorflow as tf
import tflearn
import numpy as np
import codecs
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

def aiml(query):
    intents = json.load(codecs.open('./intents.json', 'r', 'utf-8-sig'))
    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    # remove duplicates
    classes = sorted(list(set(classes)))

    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique stemmed words", words)

    # create our training data
    training = []
    # output = []
    # create an empty array for our output
    output_empty = [0] * len(classes)

    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)

    # create train and test lists
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    ops.reset_default_graph()
    
    # build neural network
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)

    # define model and setup tensorboard
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    
    # checks if model is saved 
    if(path.isdir('tflearn_logs')):
        model.load("model.tflearn")
    else:
        model.fit(train_x, train_y, n_epoch=1000,
                  batch_size=8, show_metric=True)
        model.save('model.tflearn')

    results = model.predict([bag_of_words(query, words)])
    results_index = np.argmax(results)
    tag = classes[results_index]

    for tg in intents["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
            
    # checking for unknown query       
    if (np.argmax(results) <= 47):
        return ("Try Again")

    return (random.choice(responses))

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bag_of_words(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % w)
    return(np.array(bag))
