#this script saves images of the distribution of intraclass euclidean distances calculated between shapelet feature vectors
#this is done ONCE training completes, the aim being to show the (small) deviation of the distributions between normal 
# and adversarial data

# Author: Mubarak Abdu-Aguye <maguye007.reloaded@gmail.com>
# License: BSD-3-Clause

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tslearn.shapelets import ShapeletModel, grabocka_params_to_shapelet_size_dict
from dataloader_custom import UCRTSLoader

datasetPath = './Data'

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

def featureExtract(sample):
    sample_ = sample.ravel()
    return sample_

if __name__ == "__main__":
    #Consider say the ham Dataset
    dataset = 'Ham'
    #next, we want to load the normal samples (test)

    trainLoader = UCRTSLoader(os.path.join(datasetPath, '{}'.format(dataset),'{}_TRAIN'.format(dataset)))
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
                            max_iter=1000,#50 epochs by default, 500 is good
                            verbose=1,
                            batch_size=128) 
    
    #Train the LTS classifier on the training set, which is composed of normal (i.e., unperturbed) time series.
    shp_clf.fit(train_features, train_labels)

    testLoader = UCRTSLoader(os.path.join(datasetPath, '{}'.format(dataset),'{}_TEST'.format(dataset)))
    test_features, test_labels = [],[]
    for j in testLoader:
        test_features.append(featureExtract(j[0]))
        test_labels.append(j[1]) #for normal samples

    test_features = np.array(test_features) 
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

    #convert to shapelet vectors
    normal_test_c1 = shp_clf.transform(normal_test_c1)
    normal_test_c2 = shp_clf.transform(normal_test_c2)
    adv_test_c1 = shp_clf.transform(adv_test_c1)
    adv_test_c1_f = shp_clf.transform(adv_test_c1_f)
    adv_test_c2 = shp_clf.transform(adv_test_c2)

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

    #uncomment the below lines to save the plot data
    # export = np.zeros((len(bin_edges) - 1, 3))
    # export[:,0] = sc_hist
    # export[:,1] = nadsc_hist_f
    # export[:,2] = nadsc_hist
    #np.savetxt('fig2_left.csv', export, delimiter=",", fmt="%.1f",header='Normal,FGSM,BIM')

    plt.plot(sc_hist,linewidth=2)
    plt.plot(nadsc_hist,linewidth=2)
    plt.plot(nadsc_hist_f,linewidth=2)
    plt.grid(which='both',axis='both')
    plt.legend(['Normal','BIM','FGSM'],loc=2, fontsize='xx-large')
    plt.xlabel('Euclidean Distance',fontsize=18)
    plt.ylabel('Number of Samples',fontsize=18)
    plt.show()
    