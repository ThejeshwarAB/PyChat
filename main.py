#video 1

#importing necessary libraries
import nltk
import numpy 
import tflearn
import tensorflow
import random 
import pickle
import codecs
import json
import os

#loading stemmer library 
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#reading json file

# with open("intents.json") as file:
	#data is the dataset 
	# data = json.load(file) 

data = json.load(codecs.open('intents.json', 'r', 'utf-8-sig'))
nltk.download('punkt')
words = [] #tokenized sentences 
label = [] #set of tags dataset  
doc_x = [] #has the pattern val
doc_y = [] #has the intent tags

#checking for data pickling 
try:
	with open("data.pickle", "rb") as f:
		words, label, training, outcomes = pickle.load(f) 

except:

	#reading through the json file
	for intent in data["intents"]:

		for pattern in intent["patterns"]:
			#tokenization of sentences
			wrds = nltk.word_tokenize(pattern)
			words.extend(wrds)
			doc_x.append(pattern)
			doc_y.append(intent["tag"])

		if intent["tag"] not in label:
			label.append(intent["tag"])

	#video 2

	#stemming words and sorting them
	words = [stemmer.stem(w.lower()) for w in words]
	words = sorted(list(set(words)))
	#label is also sorted
	label = sorted(label)

	training = [] #userinput 
	outcomes = [] #responses 

	out_empty = [0 for _ in range(len(label))]

	for x, doc in enumerate(doc_x):
		#bag of words
		pack = [] 

		#userinput classified
		wrds = [stemmer.stem(w) for w in doc]

		#one-hot encoding
		for w in words:
			if w in wrds:
				pack.append(1)
			else:
				pack.append(0)

		#only int not string in dnn
		output_row = out_empty[:]
		output_row[label.index(doc_y[x])] = 1

		training.append(pack)
		outcomes.append(output_row)

	training = numpy.array(training) #preprocessed data of userinput
	outcomes = numpy.array(outcomes) #preprocessed data of responses

	with open("data.pickle", "wb") as f:
		pickle.dump((words, label, training, outcomes), f)


#video 3

#tensorflow model is created here
from tensorflow.python.framework import ops
ops.reset_default_graph()
# tensorflow.reset_default_graph()

#neural network with 3 phases for training
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net ,len(outcomes[0]), activation="softmax")
net = tflearn.regression(net)

#model is initisalised
model = tflearn.DNN(net)

#saving/loading the model
try:
	model.load("model.tflearn")
except:
	model.fit(training, outcomes, n_epoch=1000, batch_size=8, show_metric= True)
	model.save("model.tflearn")

#video 4

#used for pattern matching
def pack_words(s, words):
	pack = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				pack[i] = 1

	return numpy.array(pack) 

#used for user interaction 
def chat():
	print("Start(q to quit)")
	while True:
		query = input("Type:")
		
		if query.lower() == "q":
			break

		#results of matching score 
		results = model.predict([pack_words(query,words)])
		res_max = numpy.argmax(results) 

		tag = label[res_max]

		for tg in data["intents"]:
			if tg["tag"] == tag:
				responses = tg["responses"]

		print(random.choice(responses))

chat()