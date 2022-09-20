"""
@Yuan_Wei
can help me, create functions that takes in a image path/string, example predictIMG("image.jpg")
and predict and give the apove response layout.

example of return 
    return {"label":"skin rash", "accuracy":85}
example of unkown return:
    return {"label":"unknown", "accuracy":0}

@Kenneth
can help me, create functions that takes in a strings of text, example predictTXT("This is a massage for the chatbot")
and predict and give the apove response layout.

example of return 
    return {"tag":"for tag", "response":"This is a response", "accuracy":70}

example of unkown return:
    return {"tag":"known", "response":"known", "accuracy":0}

@Zhi_Xin
The backend uses api calls. examples:
-- Image rec:
    POST http://<IP/domain>/image
    IMPORTANT NOTE: the image file needs to be sent over as FORM-DATA (multipart/form-data) with the key being called image.

    reponse will be in json:
    example of image reg:
    {"label":"unknown", "accuracy":0}

-- chatbot
    POST http://<IP/domain>/chatbot
    response will be in json:
    {"tag":"unknown", "responses":"search could not accuratly detect input contents", "accuracy":0}
"""


from flask import Flask , request , json

import os
import cv2
import numpy as np
import keras
import random
from imutils import paths
from keras.preprocessing import image
from keras.utils.image_utils import img_to_array  #Changed
from keras.applications import vgg16
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


imag_cat_size = 3  #number of categories to detec.
Imag_EPOCHS = 100   #EPOCH
Chat_EPOCHS = 500   #EPOCH value for chatbot
dataset_path = "images"
IMAGE_DIMS = (224, 224, 3)
BS = 32
print("keras version %s"%keras.__version__)
print("opencv version %s"%cv2.__version__)
lb = LabelBinarizer()

# initialize the data and labels
data = []
labels = []

#__________________________ ChatBot: load model or traign if model missing.  ______________________________________________________________________
try: 
    model = load_model('chatbot_model.h5')
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl','rb'))
    classes = pickle.load(open('classes.pkl','rb'))
except:
    words=[]
    classes = []
    documents = []
    ignore_words = ['?', '!']
    data_file = open('intents.json').read()
    intents = json.loads(data_file)


    for intent in intents['intents']:
        for pattern in intent['patterns']:

            #tokenize each word
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            #add documents in the corpus
            documents.append((w, intent['tag']))

            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # lemmaztize and lower each word and remove duplicates
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    # sort classes
    classes = sorted(list(set(classes)))
    # documents = combination between patterns and intents
    print (len(documents), "documents")
    # classes = intents
    print (len(classes), "classes", classes)
    # words = all words, vocabulary
    print (len(words), "unique lemmatized words", words)


    pickle.dump(words,open('words.pkl','wb'))
    pickle.dump(classes,open('classes.pkl','wb'))

    # create our training data
    training = []
    # create an empty array for our output
    output_empty = [0] * len(classes)
    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # lemmatize each word - create base word, in attempt to represent related words
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        # create our bag of words array with 1, if word match found in current pattern
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        
        # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        
        training.append([bag, output_row])
    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)
    # create train and test lists. X - patterns, Y - intents
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    print("Training data created")


    # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of intents to predict output intent with softmax
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #fitting and saving the model 
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=Chat_EPOCHS, batch_size=5, verbose=1)
    model.save('chatbot_model.h5', hist)

    print("model created and saved")
# ___________________________ Chatbot: Define functions _____________________________________________________________________________________________

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    # ERROR_THRESHOLD = 0.25  # Removed becoues if false cause error!!!
    # results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results = [[i,r] for i,r in enumerate(res)]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

#__________________________ Image Reconition: load model or traign if model missing. _________________________________
try:
    network = keras.models.load_model("my_model.h5")

    with open("labels.pickle", "rb") as f:
        labels = pickle.load(f)
    
    # binarize the labels
    labels = lb.fit_transform(labels)
except:
    # grab the image paths and randomly shuffle them
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images(dataset_path)))
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for ind, imagePath in enumerate(imagePaths):
        # load the image, pre-process it, and store it in the data list
        print(ind, imagePath, imagePath.split(os.path.sep)[-2])
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)
    
        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)


    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    print("[INFO] data matrix: {:.2f}MB".format(
        data.nbytes / (1024 * 1000.0)))

    with open("labels.pickle", "wb") as f:
        pickle.dump((labels), f)

    # binarize the labels
    labels = lb.fit_transform(labels)

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
        labels, test_size=0.2, random_state=42)

    # Compile and trains models
    print("[INFO] compiling model...")

    base_layers = vgg16.VGG16(include_top=False, input_shape=IMAGE_DIMS)

    for layer in base_layers.layers:
        layer.trainable = False
    
    network = models.Sequential(base_layers.layers)
    network.add(layers.Flatten())
    network.add(layers.Dense(imag_cat_size, activation="softmax"))

    network.compile(optimizer="rmsprop",
                    loss="categorical_crossentropy", metrics=['accuracy'])

    network.summary()

    #Train and evaluate models:
    history = network.fit(trainX, trainY, epochs=Imag_EPOCHS, batch_size=32, verbose=2)
    test_loss, test_acc = network.evaluate(trainX, trainY, verbose=2)

    print('test-acc', test_acc)

    network.save("my_model.h5")

#_________________________________________________ API ______________________________________________________________________________
api = Flask(__name__)

@api.route('/image' , methods=['POST'])
def image_rec():
    # print(request.files , file=sys.stderr)
    # Get Image and save it as image.jpg. could not find a way to pass it diractly to cv2.
    image = request.files.get("image")
    image.save("image.jpg")
    
    # Prediction:
    img = cv2.imread("image.jpg")
    image_tar = cv2.resize(img, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
    image_tar = image_tar.astype("float") / 255.0
    image_tar = img_to_array(image_tar)
    image_tar = np.expand_dims(image_tar, axis=0)
    proba = network.predict(image_tar)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    acc = proba[idx]

    reply = {"Identified": label}

    return json.dumps(reply)
    
@api.route('/chatbot/<msg>' , methods=['GET'])
def chatbot(msg):
    reply = {"Identified": chatbot_response(msg)}

    return json.dumps(reply)

if __name__ == '__main__':
    api.run() 