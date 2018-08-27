import pickle
from sklearn.neighbors import KNeighborsClassifier
# from MFCC_Utils import * 

data = []
noisy = []
studio = []
noisy_test = []
noisy_train = []
studio_test = []
studio_train = []
kNN = KNeighborsClassifier(5)
RATIO = 0.8


def initialize():
    global noisy, studio    
    with open('C:/Users/LENOVO-PC/Desktop/AIML/AIML-Labs-BLR/Hackathon 1/speech_data.dump', 'rb') as file_dump:
        data = pickle.load(file_dump)
    noisy += [sample for sample in data if '_n_' in sample['filename']]
    studio += [sample for sample in data if '_n_' not in sample['filename']]

def split_test_train():
    global noisy_train, noisy_test, studio_train, studio_test, test_set, test_labels
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
    train_set = [sample['features'] for sample in studio_train]
    train_labels = [sample['label'] for sample in studio_train]
    test_set = [sample['features'] for sample in studio_test]
    test_labels = [sample['label'] for sample in studio_test]
    kNN.fit(train_set, train_labels)
    # print(noisy_train[-1])
    # print(studio_test[-1])

# print(noisy_test[0]['features'].tolist())
# print(noisy_train[0]['features'].tolist())
# print(len(noisy_test), len(noisy_train), len(noisy))

######################  Studio #############################

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
def returnLabel(test_set):
    predictedLabels = kNN.predict(test_set)
    return predictedLabels[0]
# print(findAccuracy(train_set, train_labels, test_set, test_labels))
    
initialize()
split_test_train()
print(returnLabel([test_set[0]]))