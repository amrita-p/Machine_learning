import os
import glob
import pickle
import Record_audio
from sklearn.neighbors import KNeighborsClassifier

HOME_FOLDER="/Users/saratchandra/Play/AiMlTalentSprint/AIML-Labs-BLR/Hackathon"
glob_2 = "{}/Noisy_Speech/{}_*.wav"
data = []
kNN = KNeighborsClassifier(3)
test_data = []

def initialize():
    global data, kNN
    with open('speech_data_with_noisy_team.dump', 'rb') as file_dump:
        data = pickle.load(file_dump)
    train_data = [sample['features'] for sample in data]
    train_labels = [sample['label'] for sample in data]
    kNN.fit(train_data, train_labels)

def predict(test_features):
    global kNN
    absolute = kNN.predict(test_features)
    print("predict with: ", absolute)
    result = kNN.predict_proba(test_features)
    print("with confidence: ", result)
    max = 0
    index = 0
    for i, conf in enumerate(result[0]):
        if conf > max:
            max = conf
            index = i
    return index, max

def files_with_label(label, glob_pattern):
    filled_pattern = glob_pattern.format(HOME_FOLDER, label)
    return glob.glob(filled_pattern)

initialize()

for label in range(12):
    for file in files_with_label(label, glob_2):
        features = Record_audio.get_features(file)
        test_data.append({
            'features': features,
            'filename': os.path.basename(file),
            'label': label
        })
        print("Done for {}".format(file))

accurate = 0
total = 0

for sample in test_data:
    total += 1
    label, confidence = predict([sample['features']])
    if (label == sample['label']):
        accurate += 1

print(accurate * 100 / total)


# print(predict([data[-1]['features']]))
# noisy = [sample for sample in data if '_n_' in sample['filename']]
# studio = [sample for sample in data if '_n_' not in sample['filename']]

# noisy_test = []
# noisy_train = []
# studio_test = []
# studio_train = []

# RATIO = 0.8

# for label in range(12):
#     noisy_samples = [sample for sample in noisy if sample['label'] == label]
#     studio_samples = [sample for sample in studio if sample['label'] == label]
#     # print(noisy_samples[0])
#     # print(studio_samples[0])
#     noisy_index = int(RATIO*len(noisy_samples))
#     studio_index = int(RATIO*len(studio_samples))
#     noisy_train+= noisy_samples[:noisy_index]
#     noisy_test+= noisy_samples[noisy_index:]
#     studio_train+= studio_samples[:studio_index]
#     studio_test+= studio_samples[studio_index:]
#     # print(noisy_train[-1])
#     # print(studio_test[-1])

# # print(noisy_test[0]['features'].tolist())
# # print(noisy_train[0]['features'].tolist())
# # print(len(noisy_test), len(noisy_train), len(noisy))

# ######################  Studio #############################
# train_set = [sample['features'] for sample in studio_train]
# train_labels = [sample['label'] for sample in studio_train]
# test_set = [sample['features'] for sample in studio_test]
# test_labels = [sample['label'] for sample in studio_test]

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
# def findAccuracy(train_set, train_labels, test_set, test_labels, k=5):
#     kNN = KNeighborsClassifier(k)
#     kNN.fit(train_set, train_labels)
#     predictedLabels = kNN.predict(test_set)
#     total = 0
#     correct = 0
#     for i, label in enumerate(predictedLabels):
#         total += 1
#         if label == test_labels[i]:
#             correct += 1
#     return (correct * 100 / total)

# print(findAccuracy(train_set, train_labels, test_set, test_labels))
