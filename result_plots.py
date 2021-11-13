# Author: Mubarak Abdu-Aguye <maguye007.reloaded@gmail.com>
# License: BSD-3-Clause

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import csv


def getResults(pathToFile):
    output = {}
    with open(pathToFile) as readHandle:
        rdr = csv.DictReader(readHandle)
        for row in rdr:
            output[row['name']] = [row['resnet_ori'],row['resnet_fgsm_adv'],row['resnet_bim_adv'], row['proposed_ori'], row['proposed_fgsm_adv'], row['proposed_bim_adv'] ]    
    return output

def plotNormalAdversarial_Resnet(results, classifierName= 'ResNet'):
    #normal on X, adversarial on Y
    #extract X pairs and y pairs
    _x, _y,_y_ = [],[],[]
    for datum in results.keys():
        lineItem = results[datum]
        _x.append(float(lineItem[0]))
        _y.append(float(lineItem[1])) #1 = fgsm
        _y_.append(float(lineItem[2])) #1 = bim
    
    plt.axes().set_aspect('equal')
    plt.scatter(_x,_y,c='#98DFF8',marker='o',label='FGSM')
    plt.scatter(_x,_y_,c='#F0975A',marker='x',label='BIM')
    startPoint, endPoint = [-10,150],[-10,150]
    plt.plot(startPoint, endPoint, 'r',label=None)
    plt.xlabel('Normal Accuracy (ResNet) (%)')
    plt.ylabel('Adversarial Accuracy (ResNet) (%)')
    plt.title('Normal vs Adversarial Accuracy with {} Classifier'.format(classifierName))
    plt.xlim([-1,110])
    plt.ylim([-1,110])
    plt.legend()
    
    plt.savefig('normal-adv-resnet.png')
    plt.show()
    x_ = np.array(_x)
    y_,y__ = np.array(_y), np.array(_y_)
    fgsm_w, fgsm_d, fgsm_l = np.count_nonzero(x_ > y_), np.count_nonzero(x_ == y_), np.count_nonzero(x_ < y_)
    bim_w, bim_d, bim_l = np.count_nonzero(x_ > y__), np.count_nonzero(x_ == y__), np.count_nonzero(x_ < y__)
    print('Normal v FGSM ResNet W/D/L: {}/{}/{}'.format(fgsm_w, fgsm_d,fgsm_l))
    print('Normal v BIM ResNet W/D/L: {}/{}/{}'.format(bim_w, bim_d,bim_l))
    
    #Uncomment to save the raw plot data
    # export = np.zeros((len(x_), 3))
    # export[:,0] = x_ #normal
    # export[:,1] = y_ #FGSM
    # export[:,2] = y__ #BIM
    # np.savetxt('fig5a.csv', export, delimiter=",", fmt="%.1f",header='Normal,FGSM,BIM')

def plotNormalAdversarial_Traditional(results, classifierName):
    #normal on X, adversarial on Y
    #extract X pairs and y pairs
    _x, _y,_y_ = [],[],[]
    for datum in results.keys():
        lineItem = results[datum]
        _x.append(float(lineItem[3])) #normal
        _y.append(float(lineItem[4])) #fgsm
        _y_.append(float(lineItem[5])) #bim
    
    plt.axes().set_aspect('equal')
    plt.scatter(_x,_y,c='#98DFF8',marker='o',label='FGSM')
    plt.scatter(_x,_y_,c='#F0975A',marker='x',label='BIM')
    startPoint, endPoint = [-10,150],[-10,150]
    plt.plot(startPoint, endPoint, 'r',label=None)
    plt.xlabel('Normal Accuracy {} (%)'.format(classifierName))
    plt.ylabel('Adversarial Accuracy (%)'.format(classifierName))
    plt.title('Normal vs Adversarial Accuracy with {} Classifier'.format(classifierName))
    plt.xlim([-1,110])
    plt.ylim([-1,110])
    plt.legend()
    plt.savefig('normal-adv-{}.png'.format(classifierName))
    plt.show()
    x_ = np.array(_x)
    y_,y__ = np.array(_y), np.array(_y_)
    # fgsm_w, fgsm_d, fgsm_l = np.count_nonzero(x_ > y_), np.count_nonzero(x_ == y_), np.count_nonzero(x_ < y_)
    # bim_w, bim_d, bim_l = np.count_nonzero(x_ > y__), np.count_nonzero(x_ == y__), np.count_nonzero(x_ < y__)
    fgsm_w, fgsm_d, fgsm_l = np.count_nonzero((x_ - y_) > 2.5), np.count_nonzero(np.abs(x_ - y_) <=2.5), np.count_nonzero((x_ - y_) < -2.5)
    bim_w, bim_d, bim_l = np.count_nonzero((x_ - y__) > 2.5), np.count_nonzero(np.abs(x_ - y__) <= 2.5), np.count_nonzero((x_ - y__) < -2.5)
    print('Normal v FGSM Trad W/D/L: {}/{}/{}'.format(fgsm_w, fgsm_d,fgsm_l))
    print('Normal v BIM Trad W/D/L: {}/{}/{}'.format(bim_w, bim_d,bim_l))
    
    #Uncomment to save raw plot data
    # export = np.zeros((len(x_), 3))
    # export[:,0] = x_ #normal
    # export[:,1] = y_ #FGSM
    # export[:,2] = y__ #BIM
    # np.savetxt('fig5b.csv', export, delimiter=",", fmt="%.1f",header='Normal,FGSM,BIM')

def plotNormal_ResnetTraditional(results, classifierName):
    #x is normal resnet, y is normal traditional
    #extract X pairs and y pairs
    _x, _y = [],[]
    for datum in results.keys():
        lineItem = results[datum]
        _x.append(float(lineItem[0]))
        _y.append(float(lineItem[3]))
    
    plt.axes().set_aspect('equal')
    plt.scatter(_x,_y,c='#5BC421')
    startPoint, endPoint = [-10,150],[-10,150]
    plt.plot(startPoint, endPoint, 'r')
    plt.xlabel('Normal Accuracy (ResNet) (%)')
    plt.ylabel('Normal Accuracy ({}) (%)'.format(classifierName.upper()))
    plt.title('Normal Accuracy (ResNet vs {})'.format(classifierName))
    plt.xlim([-1,110])
    plt.ylim([-1,110])
    
    plt.savefig('normal-resnet-{}.png'.format(classifierName))
    plt.show()
    x_ = np.array(_x)
    y_ = np.array(_y)
    w, d, l = np.count_nonzero(x_ > y_), np.count_nonzero(x_ == y_), np.count_nonzero(x_ < y_)
    print('Normal ResNet vs Trad W/D/L: {}/{}/{}'.format(w, d,l))
    
    #Uncomment to save raw plot data
    # export = np.zeros((len(x_), 2))
    # export[:,0] = x_ #normal Resnet
    # export[:,1] = y_ #normal trad
    # np.savetxt('fig8.csv', export, delimiter=",", fmt="%.1f",header='Normal ResNet, Normal Traditional')

def plotAdversarial_ResnetTraditional(results, classifierName):
    #x is adversarial resnet, y is adversarial traditional
    #extract X pairs and y pairs
    _x, _y = [],[] #fgsms
    _x_,_y_ = [],[] #bims
    for datum in results.keys():
        lineItem = results[datum]
        _x.append(float(lineItem[1])) #resnet fgsm
        _y.append(float(lineItem[4])) #traditional fgsm
        _x_.append(float(lineItem[2])) #resnet bim
        _y_.append(float(lineItem[5])) #traditional bim
    
    plt.axes().set_aspect('equal')
    plt.scatter(_x,_y,c='#98DFF8',marker='o',label='FGSM')
    plt.scatter(_x_,_y_,c='#F0975A',marker='x',label='BIM')
    startPoint, endPoint = [-10,150],[-10,150]
    plt.plot(startPoint, endPoint, 'r',label=None)
    plt.xlabel('Adversarial Accuracy (ResNet) (%)')
    plt.ylabel('Adversarial Accuracy ({}) (%)'.format(classifierName.upper()))
    plt.title('Adversarial Accuracy (ResNet vs {})'.format(classifierName))
    plt.xlim([-1,110])
    plt.ylim([-1,110])
    plt.legend()
    plt.savefig('adv-resnet-{}.png'.format(classifierName))
    plt.show()
    x_, x__ = np.array(_x), np.array(_x_)
    y_,y__ = np.array(_y), np.array(_y_)
    fgsm_w, fgsm_d, fgsm_l = np.count_nonzero(x_ > y_), np.count_nonzero(x_ == y_), np.count_nonzero(x_ < y_)
    bim_w, bim_d, bim_l = np.count_nonzero(x__ > y__), np.count_nonzero(x__ == y__), np.count_nonzero(x__ < y__)
    print('Resnet FGSM v Trad FGSM W/D/L: {}/{}/{}'.format(fgsm_w, fgsm_d,fgsm_l))
    print('ResNet BIM v Trad BIM W/D/L: {}/{}/{}'.format(bim_w, bim_d,bim_l))
    
    #Uncomment to save raw plot data
    # export = np.zeros((len(x_), 3))
    # export[:,0] = x_ #normal
    # export[:,1] = y_ #FGSM
    # export[:,2] = y__ #BIM
    # np.savetxt('fig6.csv', export, delimiter=",", fmt="%.1f",header='Normal,FGSM,BIM')

if __name__ == "__main__":
    argv = sys.argv
    argc = len(argv)
    if argc < 3:
        print("Usage: <script.py> <name of result CSV file> <plot number: [0...4]>")
        exit()
    
    proposedFile = argv[1]
    plotType = int(argv[2])

    #get the results
    results = getResults(proposedFile)

    #extract the classifier name
    classifierName = os.path.basename(proposedFile).split('-')[0]
    #now, figure out which type of plot we want based on the choice entered
    if plotType == 1:
        plotNormalAdversarial_Resnet(results, 'ResNet')
    elif plotType == 2:
        plotNormalAdversarial_Traditional(results, classifierName)
    elif plotType == 3:
        plotNormal_ResnetTraditional(results, classifierName)
    elif plotType == 4:
        plotAdversarial_ResnetTraditional(results, classifierName)
    elif plotType == 0:
        plotNormalAdversarial_Resnet(results, 'ResNet')
        plotNormalAdversarial_Traditional(results, classifierName)
        plotNormal_ResnetTraditional(results, classifierName)
        plotAdversarial_ResnetTraditional(results, classifierName)

    print('Fin.')