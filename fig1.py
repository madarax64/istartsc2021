# Author: Mubarak Abdu-Aguye <maguye007.reloaded@gmail.com>
# License: BSD-3-Clause

#This script plots a normal vs adversarial sample for a specified dataset (Figure 1)
import os
import numpy as np
import matplotlib.pyplot as plt
from dataloader_custom import UCRTSLoader

datasetPath = './Data'

def featureExtract(sample):
    sample_ = sample.ravel()
    return sample_

if __name__ == "__main__":
    #Consider say the ham Dataset
    dataset = 'Ham'
    #next, we want to load the normal samples (test)

    trainLoader = UCRTSLoader(os.path.join(datasetPath, '{}'.format(dataset),'{}_TRAIN'.format(dataset)))
    n_classes  = trainLoader.n_classes

    testLoader = UCRTSLoader(os.path.join(datasetPath, '{}'.format(dataset),'{}_TEST'.format(dataset)))
    test_features, test_labels = [],[]
    for j in testLoader:
        test_features.append(featureExtract(j[0]))
        test_labels.append(j[1]) #for normal samples

    test_features = np.array(test_features) #todo: Normalize time series?
    test_labels = np.array(test_labels)

    advLoader = UCRTSLoader(os.path.join(datasetPath, '{}'.format(dataset),'{}-adv_BIM'.format(dataset)))
    adv_features, adv_labels = [], []
    for k in advLoader:
        adv_features.append(featureExtract(k[0]))
        adv_labels.append(k[1])
    
    adv_features = np.array(adv_features)
    adv_labels = np.array(adv_labels)

    advLoader_f = UCRTSLoader(os.path.join(datasetPath, '{}'.format(dataset),'{}-adv_FGSM'.format(dataset)))
    adv_features_f, adv_labels_f = [], []
    for l in advLoader_f:
        adv_features_f.append(featureExtract(l[0]))
        adv_labels_f.append(l[1])
    
    adv_features_f = np.array(adv_features_f)
    adv_labels_f = np.array(adv_labels_f)

    index = 0
    normal = test_features[index]
    adversarial = adv_features[index]
    length = len(normal)

    plt.plot(list(range(length)),normal)
    plt.plot(list(range(length)),adversarial)
    plt.legend(['Original Signal','Perturbed Signal'])
    plt.xlabel('Time [frame]')
    plt.ylabel('Signal alue')
    plt.show()
    
    #and now, export this as one happy matrix
    export = np.zeros((length, 2))
    export[:,0] = normal
    export[:,1] = adversarial
    np.savetxt('fig1.csv', export, delimiter=",", fmt="%.2f",header='Normal,Adversarial')
    #exit()

    