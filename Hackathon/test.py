# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 13:34:23 2018

@author: Amrita
"""
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#from MFCC_Utils import *


#with open('speech_data.dump', 'rb') as file_dump:
#    data = pickle.load(file_dump)

#print(data[0])
#noisy = np.array([sample for sample in data if '_n_' in sample['filename']])
#print(type(noisy))



#from sklearn.neighbors import KNeighborsClassifier
 

data = []

with open('speech_data.dump', 'rb') as file_dump:
    data = pickle.load(file_dump)

# print(data[0])

classLabels = list(range(12))


noisy = np.array([sample for sample in data if '_n_' in sample['filename']])
studio = np.array([sample for sample in data if '_n_' not in sample['filename']])

noisy_test = []
noisy_train = []
studio_test = []
studio_train = []

#print(noisy[0]['label'])

RATIO = 0.8
#import librosa

# Visualization

#import librosa.display



for label in range(12):
    noisy_samples = [sample for sample in noisy if sample['label'] == label]
    studio_samples = [sample for sample in studio if sample['label'] == label]
    # print(noisy_samples[0])
    # print(studio_samples[0])
    noisy_index = int(RATIO*len(noisy_samples))
    studio_index = int(RATIO*len(studio_samples))
    noisy_train+= noisy_samples[:noisy_index]
    noisy_test+= noisy_samples[noisy_index:]
    studio_train+= studio_samples[:studio_index]
    studio_test+= studio_samples[studio_index:]
    # print(noisy_train[-1])
    # print(studio_test[-1])

# print(noisy_test[0]['features'].tolist())
# print(noisy_train[0]['features'].tolist())
# print(len(noisy_test), len(noisy_train), len(noisy))
#log_S = librosa.power_to_db(S, ref=np.max)

######################  Studio #############################
train_set = np.array([sample['features']/100 for sample in studio_train])
train_labels = np.array([sample['label'] for sample in studio_train])
test_set = np.array([sample['features']/100 for sample in studio_test])
test_labels = np.array([sample['label'] for sample in studio_test])

print(test_set[0])

#mfcc_test = librosa.feature.mfcc(test_set, n_mfcc=50)

# Find 1st order delta_mfcc
#delta1_mfcc_train = librosa.feature.delta(mfcc_train, order=1)

# Find 2nd order delta_mfcc
#delta2_mfcc_test = librosa.feature.delta(mfcc_test, order=2)


#S = librosa.feature.melspectrogram(train_set, sr=, hop_length=int(0.020*sample_rate), n_mels=128)
#print(S.shape)
######################  Noisy #############################
# train_set = [sample['features'] for sample in noisy_train]
# train_labels = [sample['label'] for sample in noisy_train]
# test_set = [sample['features'] for sample in noisy_test]
# test_labels = [sample['label'] for sample in noisy_test]

###################### All sample #############################
# train_set = [sample['features'] for sample in noisy_train + studio_train]
# train_labels = [sample['label'] for sample in noisy_train + studio_train]
# test_set = [sample['features'] for sample in noisy_test + studio_test]
# test_labels = [sample['label'] for sample in noisy_test + studio_test]


# print(predict_and_find_accuracy('linear', np.array(train_set), train_labels, np.array(test_set), test_labels))
def findAccuracy(train_set, train_labels, test_set, test_labels, k=5):
    kNN = KNeighborsClassifier(k)
    kNN.fit(train_set, train_labels)
    predictedLabels = kNN.predict(test_set)
    total = 0
    correct = 0
    for i, label in enumerate(predictedLabels):
        total += 1
        if label == test_labels[i]:
            correct += 1
    return (correct * 100 / total)
#def accuracy(k):
#    global predict_and_find_accuracy
#    acc = predict_and_find_accuracy("kNN", train_set, train_labels, test_set, test_labels, k=5)
 #   return acc

print(findAccuracy(train_set,train_labels,test_set,test_labels))