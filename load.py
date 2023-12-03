
import numpy as np
import streamlit as st
import os
from tqdm import tqdm
from PIL import Image
import nltk
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
import pydot,graphviz



from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input,Dropout,Dense,Embedding,LSTM,add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))





file_path = os.path.join('captions.txt')

with open(file_path, 'r') as f:
    next(f)  # Skip the first line if you want
    # for line in f:
    #     print(line.strip())
    captions_doc = f.read()

# create mapping of image to captions
mapping = {}
for line in tqdm(captions_doc.split('\n')):
    #split the line by comma
    tokens = line.split(',')
    if len(line)<2:
        continue
    image_id,caption = tokens[0],tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    #convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    #store the caption
    mapping[image_id].append(caption)

def clean(mapping):
    for  key,captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            #preprocessing steps
            #convert to lowercase
            #delete digits, special chars
            #delete  additional spaces

            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]','')
            caption = caption.replace('\s+',' ')
            #add start and end tags to the caption
            caption = '<start>'+" ".join([word for word in caption.split() if len(word)>1]) +'<end>'
            captions[i] = caption
#before preprocess of text
# print(mapping['1000268201_693b08cb0e'])
#preprocess the text
clean(mapping)
#after preprocess of text
# print(mapping['1000268201_693b08cb0e'])
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)
# print(len(all_captions))

# print(all_captions[:10])

#tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index)+1
# print(vocab_size)

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
# print(max_length)


## Train Test Split
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]



import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# def data_generator(data_keys, mapping, feature_list, tokenizer, max_length, vocab_size, batch_size):
#     while True:
#         np.random.shuffle(data_keys)  # Shuffle data keys at the beginning of each epoch
#         X1, X2, y = list(), list(), list()
#         for key in data_keys:
#             captions = mapping[key]
#             for caption in captions:
#                 seq = tokenizer.texts_to_sequences([caption])[0]
#                 for i in range(1, len(seq)):
#                     in_seq, out_seq = seq[:i], seq[i]
#                     in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
#                     out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
#                     X1.append(feature_list[key][0])
#
#                     X2.append(in_seq)
#                     y.append(out_seq)
#                     if len(X1) == batch_size:
#                         yield ([np.array(X1), np.array(X2)], np.array(y))
#                         X1, X2, y = list(), list(), list()
def data_generator(data_keys, mapping, feature_list, tokenizer, max_length, vocab_size, batch_size):
    while True:
        np.random.shuffle(data_keys)  # Shuffle data keys at the beginning of each epoch
        X1, X2, y = list(), list(), list()
        for key in data_keys:
            # Attempt to convert key to integer
            try:
                key_int = int(key)
            except ValueError:
                # Skip this instance if key cannot be converted to integer
                continue

            if key_int < 0 or key_int >= len(feature_list):
                # Skip this instance if key_int is out of bounds for feature_list
                continue

            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    # Access the first element of feature_list[key_int]
                    X1.append(feature_list[key_int][0])
                    X2.append(in_seq)
                    y.append(out_seq)

                    if len(X1) == batch_size:
                        print("Yielding batch")
                        yield ([np.array(X1), np.array(X2)], np.array(y))
                        X1, X2, y = list(), list(), list()

##MODEL CREATION
# encoder model
# image feature layers
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# plot the model
# plot_model(model, show_shapes=True)

# train the model
epochs = 20
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    # create data generator
    generator = data_generator(train, mapping, feature_list, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text


from nltk.translate.bleu_score import corpus_bleu

# validate with test data
actual, predicted = list(), list()

for key in tqdm(test):
    # get actual caption
    captions = mapping[key]
    # predict the caption for image
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)

# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
generate_caption("1001773457_577c3a7d70.jpg")