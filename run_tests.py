# Author: Mubarak Abdu-Aguye <maguye007.reloaded@gmail.com>
# License: BSD-3-Clause

#This script runs the classifier scripts on the specified range of datasets, one by one
#It sets up the scaffold for the data and result folders beforehand
import os
import sys
from subprocess import run

startDataset = '50words'
endDataset = 'yoga'
prohibitedDatasets = []
pathToData = "./Data/"
pathToResults = "./Results/"

executableName = 'python3' if os.name == "posix" else "python"
n_runs = 10
classifierScripts = ['Hypothesis 3/rocket_classifier.py', 'Hypothesis 3/learning_shapelets.py']

if __name__ == "__main__":
    if not os.path.isdir(pathToResults):
        os.mkdir(pathToResults)
    
    datasets = sorted(os.listdir(pathToData))
    startIndex = datasets.index(startDataset)
    endIndex = datasets.index(endDataset)
    for i, d in enumerate(datasets):
        if i < startIndex:
            print('Seeking past dataset {}...'.format(d))
            continue
        if i > endIndex:
            exit()
        
        if d in prohibitedDatasets:
            print('Skipping prohibited dataset {}..'.format(d))
            continue
            
        #now we're here. Construct the path to the dataset and bootstrap the classifier appropriately
        datasetPath = os.path.join(pathToData,d)
        print('Now processing {}..'.format(d))
        for script in classifierScripts:
            for n in range(n_runs):
                print('\tRun {} of {} of {}..'.format(n + 1,n_runs,script))
                run([executableName, script, d, pathToResults])
            print('Completed runs for {}..'.format(script))
    
    print('Fin')


