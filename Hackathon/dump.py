#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 10:37:04 2018

@author: saratchandra
"""

import glob
import Record_audio
import sklearn
import pickle
import os

classLabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

HOME_FOLDER="/Users/saratchandra/Play/AiMlTalentSprint/AIML-Labs-BLR/Hackathon"
# glob_1 = "{}/speech_data/{}_*.wav"
glob_2 = "{}/Noisy_Speech/{}_*.wav"

def files_with_label(label, glob_pattern):
    filled_pattern = glob_pattern.format(HOME_FOLDER, label)
    return glob.glob(filled_pattern)

data = []

with open('speech_data.dump', 'rb') as current_pikle:
    data = pickle.load(current_pikle)

print('Before adding new samples: ', len(data))

for label in classLabels:
    for file in files_with_label(label, glob_2):
        features = Record_audio.get_features(file)
        data.append({
            'features': features,
            'filename': os.path.basename(file),
            'label': label
        })
        print("Done for {}".format(file))

with open('speech_data_with_noisy_team.dump', 'wb') as dump_file:
    pickle.dump(data, dump_file)

print('After adding new samples: ', len(data))