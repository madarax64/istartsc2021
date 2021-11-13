# Author: Mubarak Abdu-Aguye <maguye007.reloaded@gmail.com>
# License: BSD-3-Clause

from sklearn.metrics import accuracy_score
from dataloader_custom import UCRTSLoader
import numpy as np
from pyts.transformation import ROCKET
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sys
import os

dataPath = "./Data"

def featureExtract(sample):
    sample_ = sample.ravel()
    return sample_

if __name__ == "__main__":
    argv = sys.argv
    argc = len(argv)
    if argc < 3:
        print("Usage: <script.py> <path to dataset> <path to results>")
        exit()
    #for some dataset
    dataset = argv[1]
    resultPath = argv[2]
    
    datasetPath = os.path.join(dataPath, dataset)

    #redirect output appropriately
    resultHandle = open(os.path.join(resultPath, 'ROCKET_{}_results.txt'.format(dataset)),'a+')
    sys.stdout = resultHandle
    #let us load the training data
    trainLoader = UCRTSLoader(os.path.join(datasetPath, '{}_TRAIN'.format(dataset)))
    train_features, train_labels = [],[]
    for i in trainLoader:
        train_features.append(featureExtract(i[0]))
        train_labels.append(i[1])

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    transform_ = ROCKET(n_kernels=10000)
    transform_.fit(train_features, train_labels)
    train_features = transform_.transform(train_features)
    
    #classifier = RandomForestClassifier(200)
    classifier = LogisticRegression(penalty='l2')
    classifier.fit(train_features, train_labels)

    #then, let us prepare the testing + anomalous data
    testLoader = UCRTSLoader(os.path.join(datasetPath, '{}_TEST'.format(dataset)))
    test_features, test_labels = [],[]
    for j in testLoader:
        test_features.append(featureExtract(j[0]))
        test_labels.append(j[1]) #for normal samples

    test_features = np.array(test_features) 
    test_labels = np.array(test_labels)

    test_features = transform_.transform(test_features)
    test_predictions = classifier.predict(test_features)
    print('Accuracy: {}'.format(accuracy_score(test_labels, test_predictions)))
    
    advLoader = UCRTSLoader(os.path.join(datasetPath, '{}-adv_BIM'.format(dataset)))
    adv_features, adv_labels = [], []
    for k in advLoader:
        adv_features.append(featureExtract(k[0]))
        adv_labels.append(k[1])
    
    adv_features = np.array(adv_features)
    adv_labels = np.array(adv_labels)

    adv_features = transform_.transform(adv_features)
    adv_predictions = classifier.predict(adv_features)
    print('Accuracy (BIM): {}'.format(accuracy_score(adv_labels, adv_predictions)))
    
    ######################################################################################
    advLoader = UCRTSLoader(os.path.join(datasetPath, '{}-adv_FGSM'.format(dataset)))
    adv_features, adv_labels = [], []
    for k in advLoader:
        adv_features.append(featureExtract(k[0]))
        adv_labels.append(k[1])
    
    adv_features = np.array(adv_features)
    adv_labels = np.array(adv_labels)

    adv_features = transform_.transform(adv_features)
    adv_predictions = classifier.predict(adv_features)
    print('Accuracy (FGSM): {}'.format(accuracy_score(adv_labels, adv_predictions)))

    resultHandle.close()
    
