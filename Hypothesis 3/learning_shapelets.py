#Learning Shapelets Classifier

# Author: Mubarak Abdu-Aguye <maguye007.reloaded@gmail.com>
# License: BSD-3-Clause

import numpy as np
from tensorflow.keras.optimizers import Adam
import os
import sys
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
from sklearn.metrics import accuracy_score

from dataloader_custom import UCRTSLoader

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
    resultHandle = open(os.path.join(resultPath, 'LTS_{}_results.txt'.format(dataset)),'a+')
    sys.stdout = resultHandle

    trainLoader = UCRTSLoader(os.path.join(datasetPath, '{}_TRAIN'.format(dataset)))
    n_classes  = trainLoader.n_classes
    train_features, train_labels = [],[]
    for i in trainLoader:
        train_features.append(featureExtract(i[0]))
        train_labels.append(i[1])

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=train_features.shape[0],
                                                        ts_sz=train_features.shape[1],
                                                        n_classes=n_classes,
                                                        l=0.1, #default: 0.1
                                                        r=4) 

    shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                            optimizer=Adam(lr=.001),
                            weight_regularizer=0.001, #0.01
                            max_iter=2000,#50 epochs by default, 500 is good
                            verbose=0,
                            batch_size=128) #added batch size 16th feb 2020
    shp_clf.fit(train_features, train_labels)

    testLoader = UCRTSLoader(os.path.join(datasetPath, '{}_TEST'.format(dataset)))
    test_features, test_labels = [],[]
    for j in testLoader:
        test_features.append(featureExtract(j[0]))
        test_labels.append(j[1]) #for normal samples

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    predictions = shp_clf.predict(test_features)
    print("Accuracy: {}".format(accuracy_score(test_labels, predictions)))

    advLoader = UCRTSLoader(os.path.join(datasetPath, '{}-adv_BIM'.format(dataset)))
    adv_features, adv_labels = [], []
    for k in advLoader:
        adv_features.append(featureExtract(k[0]))
        adv_labels.append(k[1])
    
    adv_features = np.array(adv_features)
    adv_labels = np.array(adv_labels)

    predictions = shp_clf.predict(adv_features)
    print("Accuracy (BIM): {}".format(accuracy_score(adv_labels, predictions)))

    advLoader = UCRTSLoader(os.path.join(datasetPath, '{}-adv_FGSM'.format(dataset)))
    adv_features, adv_labels = [], []
    for k in advLoader:
        adv_features.append(featureExtract(k[0]))
        adv_labels.append(k[1])
    
    adv_features = np.array(adv_features)
    adv_labels = np.array(adv_labels)

    predictions = shp_clf.predict(adv_features)
    print("Accuracy (FGSM): {}".format(accuracy_score(adv_labels, predictions)))

    resultHandle.close()
    