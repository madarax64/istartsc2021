# Author: Mubarak Abdu-Aguye <maguye007.reloaded@gmail.com>
# License: BSD-3-Clause

#This script takes the path to the IJCNN adversarial files (in a folder) and the UCR TS Archive folder
#Then creates a new folder and puts them all in there, together

import os
import shutil
import sys

outputDir = 'Data'

if __name__ == "__main__":
    argv = sys.argv
    argc = len(argv)
    if argc < 3:
        print('Usage: <script.py> <path to ucr-ts folder> <path to adversarial files>')
        exit()
    
    pathToUA = argv[1]
    pathToAdv = os.path.join(argv[2],'results-sdm-2019')
    bimFolder = os.path.join(pathToAdv, 'bim')
    fgsmFolder = os.path.join(pathToAdv, 'fgsm')

    tscFolders = os.listdir(pathToUA)
    if not (os.path.isdir(outputDir)):
        os.mkdir(outputDir)
    
    for dataset in tscFolders:
        targetDir = os.path.join(outputDir, dataset)
        if not os.path.isdir(targetDir):
            os.mkdir(targetDir)
        trainTestSrc = os.path.join(pathToUA, dataset)
        bimSrc = os.path.join(bimFolder, '{}-adv'.format(dataset))
        fgsmSrc = os.path.join(fgsmFolder, '{}-adv'.format(dataset))
        fileEntries = [os.path.join(trainTestSrc, a) for a in os.listdir(trainTestSrc)]

        #then copy the train and test files to the target
        [shutil.copy(a,targetDir) for a in fileEntries]
        #next, copy the bim and fgsm files
        targetBimName = '{}-adv_BIM'.format(dataset)
        targetFgsmName = '{}-adv_FGSM'.format(dataset)

        shutil.copy(bimSrc, os.path.join(targetDir, targetBimName))
        shutil.copy(fgsmSrc, os.path.join(targetDir, targetFgsmName))
        print('Copied {} files successfully...'.format(dataset))
    
    print('Fin.')


