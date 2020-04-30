#Importing the libraries
import nltk
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Opening the json file, which we use
with open("intents.json") as file:
    data = json.load(file)

# Looping trough the data
try:
    # Loading existing model/file if it exist
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Ignoring "?" in patterns.
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]

    # Sort and remove duplicates
    words = sorted(list(set(words)))
    labels = sorted(labels)

# Checking the labels
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

# Checking the patterns if there is the word or there is none
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        # Checking the tag row
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # Taking the lists and changing them into arrays
    training = numpy.array(training)
    output = numpy.array(output)

    # Opening the data file
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

try:
    model.load('model.tflearn')
except:
    tensorflow.reset_default_graph()

    # Getting the probability of the word with the use of neurons in neural network
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    # Training the model
    model = tflearn.DNN(net)

    # Passing the training data and saving it
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    # Getting the list of tokenized words
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Getting the probability of input. And outputting the highest probability output.
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        # Getting the probability of the tag from intent
        tag = labels[results_index]

        # Randomly picks response of the highest probability tag
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
