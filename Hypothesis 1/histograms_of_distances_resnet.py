#this script saves images of the distribution of intraclass euclidean distances calculated between ResNet feature vectors
#this is done USING PRETRAINED MODELS, the aim being to show the deviation of the distributions between normal and adversarial
#data

# Author: Mubarak Abdu-Aguye <maguye007.reloaded@gmail.com>
# License: BSD-3-Clause

import os
import numpy as np
import matplotlib.pyplot as plt
from dataloader_custom import UCRTSLoader
import tensorflow.keras as keras

datasetPath = './Data'
clf = None

def getDistances(seqA, seqB, skipSelf=False):
    distances = []
    for i in range(len(seqA)):
        for j in range(len(seqB)):
            if (i == j and skipSelf):
                pass
            else:
                distances.append(np.sqrt(np.sum((seqA[i] - seqB[j]) ** 2)))
    
    return distances

def plotHistogram(sequence, set_title):
    plt.hist(sequence)
    plt.set_title(set_title)
    #plt.show()

def featureExtract(sample):
    sample_ = sample[np.newaxis,:]
    vector = clf.predict(sample_)
    return vector.ravel()
    
if __name__ == "__main__":
    #Consider say the ham Dataset
    dataset = 'Ham'
    #next, we want to load the normal samples (test)

    #now, load the pretrained resnet model
    model_path = os.path.join("pre-trained-resnet","{}".format(dataset),"best_model.hdf5")
    model = keras.models.load_model(model_path)
    input_ = model.get_layer('input_1').input
    bottleneck = model.get_layer('global_average_pooling1d_1').output
    detection_model = keras.Model(inputs=[input_], outputs=[bottleneck])
    clf = detection_model

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

    c1_mask = test_labels == 0
    c2_mask = ~c1_mask
    normal_test_c1 = test_features[c1_mask]
    normal_test_c2 = test_features[c2_mask]
    adv_test_c1 = adv_features[c1_mask]
    adv_test_c1_f = adv_features_f[c1_mask]
    adv_test_c2 = adv_features[c2_mask]

    same_class_distances = getDistances(normal_test_c1, normal_test_c1)
    normal_adv_sc = getDistances(normal_test_c1, adv_test_c1)
    normal_adv_sc_f = getDistances(normal_test_c1, adv_test_c1_f)
    normal_interclass = getDistances(normal_test_c1, normal_test_c2)
    cross = getDistances(normal_test_c1, adv_test_c2)

    #now, construct histograms over each
    sc_hist, bin_edges = np.histogram(same_class_distances,density=False)
    nadsc_hist, bin_edges = np.histogram(normal_adv_sc,bins=bin_edges, density=False)
    nadsc_hist_f, bin_edges = np.histogram(normal_adv_sc_f,bins=bin_edges, density=False)
    nic_hist, bin_edges = np.histogram(normal_interclass, bins=bin_edges, density=False)
    cross_hist, bin_edges = np.histogram(cross, bins=bin_edges, density=False)

    #uncomment to save the plot data
    # export = np.zeros((len(bin_edges) - 1, 3))
    # export[:,0] = sc_hist
    # export[:,1] = nadsc_hist_f
    # export[:,2] = nadsc_hist
    #np.savetxt('fig2_right.csv', export, delimiter=",", fmt="%.1f",header='Normal,FGSM,BIM')
    
    plt.plot(sc_hist, linewidth=2)
    plt.plot(nadsc_hist,linewidth=2)
    plt.plot(nadsc_hist_f, linewidth=2)
    plt.grid(which='both',axis='both')
    plt.legend(['Normal','BIM','FGSM'],loc=1,fontsize='xx-large')
    plt.xlabel('Euclidean Distance',fontsize=18)
    plt.ylabel('Number of Samples',fontsize=18)
    plt.show()
    exit()