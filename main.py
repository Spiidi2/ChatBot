# Kirjastojen importtaus
import nltk
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Avaa json tiedoston käyttöön
with open("questions.json") as file:
    data = json.load(file)

# Datan looppaus
try:
    # Lataa olemassa olevan modelin, jos semmoinen löytyy
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for questions in data["questions"]:
        for pattern in questions["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(questions["tag"])

        if questions["tag"] not in labels:
            labels.append(questions["tag"])

    # Tarkistaa inputin ja ignooraa kysymysmerkin
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]

    # Sort and remove duplicates
    words = sorted(list(set(words)))
    labels = sorted(labels)

# Tarkistaa labelit
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

# Inpustista tarkkailee mitä sanoja on ja mitä ei ole
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        # Tarkistaa mihin tägiin inputti kuuluu
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # Listasta muodostaa numeerisen sarjan
    training = numpy.array(training)
    output = numpy.array(output)

    # Avaa data tiedoston
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

try:
    model.load('model.tflearn')
except:
    tensorflow.reset_default_graph()

    # Todennäköisyydellä valitsee minkä outputin tarjoaa käyttäjälle, neural networking avulla
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    # Treenaa modelia?
    model = tflearn.DNN(net)

    # Muuttaa treenauksen data muotoon ja tallentaa sen
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    # Tarkistaa inputin sanat ja järjestää ne numero sarjaksi
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

        # Katsoo tödennäköisyyden inputeista ja valitsee todennäköisimmän outputin.
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        # Katsoo mihin tägiin inputti kuuluu todennäköisimmin
        tag = labels[results_index]

        #  Valitsee jonkun outputin valitsemastaan tägistä.
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
