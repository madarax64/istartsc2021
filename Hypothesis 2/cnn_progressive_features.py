#this script saves images of the distribution of intraclass euclidean distances calculated between ResNet feature vectors
#this is done as training proceeds, the aim being to show the deviation of the distributions between normal and adversarial
#data over time

# Author: Mubarak Abdu-Aguye <maguye007.reloaded@gmail.com>
# License: BSD-3-Clause

import numpy as np
import json
import sys
import os
import tensorflow.keras as keras
from dataloader_custom import UCRTSLoader
import matplotlib.pyplot as plt

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
dataPath = "./Data"

terminalEpochs = 200
increments = 100
clf = None

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def featureExtract(sample):
    sample_ = sample[np.newaxis,:]
    vector = clf.predict(sample_)
    return vector.ravel()

def getDistances(seqA, seqB, skipSelf=False):
    distances = []
    for i in range(len(seqA)):
        for j in range(len(seqB)):
            if (i == j and skipSelf):
                pass
            else:
                distances.append(np.sqrt(np.sum((seqA[i] - seqB[j]) ** 2)))
    
    return distances

def process(loader):
    features,labels = [], []
    for item in loader:
        features.append(item[0])
        labels.append(item[1])
    
    features = np.array(features)
    labels = np.array(labels)

    return features, labels

if __name__ == "__main__":
    argv = sys.argv
    argc = len(argv)
    if argc < 2:
        print("Usage: <script.py> <dataset e.g Ham>")
        exit()

    dataset = argv[1]
    datasetPath = os.path.join(dataPath, dataset)

    #load the training data
    trainLoader = UCRTSLoader(os.path.join(datasetPath, '{}_TRAIN'.format(dataset)))
    n_classes = trainLoader.n_classes

    train_features, train_labels = process(trainLoader)

    progressive_histograms = [] #for file persistence later

    #Now, load and retrain the model
    for te in range(0,terminalEpochs + 1,increments):
        jsonData = None
        with open('network.json','r') as readHandle:
            jsonData = readHandle.read()
        modelObj = json.loads(jsonData)
        modelObj['config']['layers'][37]['config']['units'] = n_classes #set num of output neurons
        modelObj['config']['layers'][0]['config']['batch_input_shape'] = [None, train_features.shape[1], train_features.shape[2]] #set input shape appropriately
        jsonData = json.dumps(modelObj)
        new_model = keras.models.model_from_json(jsonData)
        new_model.compile('adam','categorical_crossentropy',metrics=['acc'])

        train_labels_ = keras.utils.to_categorical(train_labels)
        _ = new_model.fit(train_features, train_labels_, epochs=te,batch_size=32, verbose=0)

        #now, visualize the features for same class, normal and adversarial data points
        input_ = new_model.get_layer('input_1').input
        bottleneck = new_model.get_layer('global_average_pooling1d_1').output
        detection_model = keras.Model(inputs=[input_], outputs=[bottleneck])
        clf = detection_model

        testLoader = UCRTSLoader(os.path.join(datasetPath, '{}_TEST'.format(dataset)))
        test_features, test_labels = [],[]
        for j in testLoader:
            test_features.append(featureExtract(j[0]))
            test_labels.append(j[1]) #for normal samples

        test_features = np.array(test_features)
        test_labels = np.array(test_labels)

        advLoader = UCRTSLoader(os.path.join(datasetPath, '{}-adv_BIM'.format(dataset)))
        adv_features, adv_labels = [], []
        for k in advLoader:
            adv_features.append(featureExtract(k[0]))
            adv_labels.append(k[1])
        
        adv_features = np.array(adv_features)
        adv_labels = np.array(adv_labels)
        
        advLoader_fgsm = UCRTSLoader(os.path.join(datasetPath, '{}-adv_FGSM'.format(dataset)))
        adv_features_f, adv_labels_f = [], []
        for l in advLoader_fgsm:
            adv_features_f.append(featureExtract(l[0]))
            adv_labels_f.append(l[1])
        
        adv_features_f = np.array(adv_features_f)
        adv_labels_f = np.array(adv_labels_f)

        c1_mask = test_labels == 0
        c2_mask = ~c1_mask
        normal_test_c1 = test_features[c1_mask]
        normal_test_c2 = test_features[c2_mask]
        adv_test_c1 = adv_features[c1_mask]
        adv_test_c2 = adv_features[c2_mask]
        adv_test_c1f = adv_features_f[c1_mask]
        adv_test_c2f = adv_features_f[c2_mask]

        same_class_distances = getDistances(normal_test_c1, normal_test_c1)
        normal_adv_sc = getDistances(normal_test_c1, adv_test_c1)
        normal_adv_sc_f = getDistances(normal_test_c1, adv_test_c1f)
        normal_interclass = getDistances(normal_test_c1, normal_test_c2)
        cross = getDistances(normal_test_c1, adv_test_c2)

        sc_hist, bin_edges1 = np.histogram(same_class_distances,density=False)
        nadsc_hist, bin_edges2 = np.histogram(normal_adv_sc, density=False)
        nadsc_hist_f, bin_edges3 = np.histogram(normal_adv_sc_f, density=False)

        maximum = -1
        maximum = bin_edges1 if max(bin_edges1) > max(bin_edges2) else bin_edges2
        maximum = maximum if max(maximum) > max(bin_edges3) else bin_edges3

        sc_hist, bin_edges1 = np.histogram(same_class_distances,bins=maximum, density=False)
        nadsc_hist, bin_edges2 = np.histogram(normal_adv_sc, bins=maximum, density=False)
        nadsc_hist_f, bin_edges3 = np.histogram(normal_adv_sc_f, bins=maximum, density=False)

        plt.plot(sc_hist)
        plt.plot(nadsc_hist)
        plt.plot(nadsc_hist_f)
        plt.grid(which='both',axis='both')
        plt.legend(['Normal','BIM','FGSM'],loc=1,fontsize='x-large')
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Number of Samples')
        plt.tight_layout()
        plt.savefig('{}.png'.format(str(te)))
        print('Saved figure for {}'.format(str(te)))
        plt.clf()

        progressive_histograms.append(sc_hist)
        progressive_histograms.append(nadsc_hist_f)
        progressive_histograms.append(nadsc_hist)

        #uncomment the below lines to save the raw data to disk
        # export = np.zeros((len(maximum) - 1, 3))
        # export[:,0] = sc_hist
        # export[:,1] = nadsc_hist_f
        # export[:,2] = nadsc_hist
        # np.savetxt('fig3_{}.csv'.format(te), export, delimiter=",", fmt="%.1f",header='Normal,FGSM,BIM')
    