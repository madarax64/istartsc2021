#This script analyzes the results, based on a classifier code and a dataset name

# Author: Mubarak Abdu-Aguye <maguye007.reloaded@gmail.com>
# License: BSD-3-Clause
import os
import sys
import numpy as np
import texttable
import csv

referenceResultsFile = "fawaz_results.csv"

def loadReferenceResults(filePath = referenceResultsFile):
    resultsDict = {}
    with open(filePath) as readHandle:
        rawFile = csv.DictReader(readHandle)
        for r in rawFile:
            resultsDict[r['']] = {'ori': r['resnet_ori'], 'fgsm': r['resnet_fgsm_adv'], 'bim': r['resnet_bim_adv']}
    return resultsDict

def obtainResult(lines):
    #now we have the lines. Parse and extract the data. We can extend the valueDict to account for new fields as necessary.
    valueDict = {"Accuracy": [],
                "Accuracy (FGSM)": [], 
                "Accuracy (BIM)": []
                }
    
    for line in lines:
        descriptor, score = line.split(': ')
        score = float(score)
        targetList = valueDict[descriptor]
        targetList.append(score)

    #now, do statistics - and store the results
    resultDict = {}
    for k in valueDict.keys():
        valueDict[k] = np.array(valueDict[k])
        mean, std = np.mean(valueDict[k]), np.std(valueDict[k])
        count = len(valueDict[k])
        #print('{}: Mean = {}, Std = {} as computed over {} runs'.format(k,mean, std, count))
        resultDict[k] = [mean, std, count] #we can use these for later
    
    return resultDict

def initializeTextTable(classifierCode = "LTS"):
    table = texttable.Texttable()
    table.set_cols_align(["c","c","c","c","c","c","c"])
    table.set_cols_width([15] * 7)
    #table.add_rows([["Classifier:",classifierCode,"",""]], header=True)
    table.add_rows([["","Reference (ResNet)","","","Classifier:\r\n {}".format(classifierCode),"",""]], header=True)
    table.add_rows([["Dataset","Normal","FGSM","BIM","Normal","FGSM","BIM"]], header=False)
    return table

def updateTextTable(table, datasetName, resultDict, referenceResults): 
    accuracyString = str(resultDict['Accuracy'][0] * 100.0)[:5] + " +/- " + str(resultDict['Accuracy'][1] * 100.0)[:5]
    fgsmString = str(resultDict['Accuracy (FGSM)'][0] * 100.0)[:5] + " +/- " + str(resultDict['Accuracy (FGSM)'][1] * 100.0)[:5]
    bimString = str(resultDict['Accuracy (BIM)'][0] * 100.0)[:5] + " +/- " + str(resultDict['Accuracy (BIM)'][1] * 100.0)[:5]
    
    
    table.add_rows([[datasetName, referenceResults['ori'], referenceResults['fgsm'], referenceResults['bim'], accuracyString,fgsmString, bimString]],header=False)
    csvString = '{},{},{},{},{},{},{}\n'.format(datasetName, referenceResults['ori'], referenceResults['fgsm'], referenceResults['bim'], str(resultDict['Accuracy'][0] * 100.0)[:5],str(resultDict['Accuracy (FGSM)'][0] * 100.0)[:5], str(resultDict['Accuracy (BIM)'][0] * 100.0)[:5])
    return table, csvString

def saveTextTableTo(table, fileName):
    with open(fileName,'w') as writeHandle:
        writeHandle.write(table.draw())

def saveLinesTo(lines, fileName, header=None):
    with open(fileName,'w') as writeHandle:
        if not header is None:
            writeHandle.write(header)
        writeHandle.writelines(lines)


if __name__ == "__main__":
    argv = sys.argv
    argc = len(argv)
    if argc < 2:
        print("Usage: <script.py> <path to results>")
        exit()

    resultPath = argv[1]
    classifierCode = input('Enter classifier code: ').strip().upper()
    datasetName = input('Enter dataset name, or * for all: ').strip()
    tt = initializeTextTable(classifierCode)
    referenceResults = loadReferenceResults()

    if datasetName == '*':
        #wildcard, so we're going to do this the texttable way
        resultFiles = sorted([a for a in os.listdir(resultPath) if a.startswith(classifierCode + '_')])
        csvStrings = []
        #then, for each data file, get the results and update the table
        for rf in resultFiles:
            rfPath = os.path.join(resultPath, rf)
            prefix = rf.split('_')[0] + "_"
            datasetName_ = rf[rf.index(prefix) + len(prefix): rf.index("_results.txt")] #pull the actual datasetname from the result file

            lines = None
            with open(rfPath) as handle:
                lines = handle.readlines()
            
            result = obtainResult(lines)
            tt, cs = updateTextTable(tt, datasetName_, result, referenceResults[datasetName_])
            csvStrings.append(cs)

        print(tt.draw())
        saveTextTableTo(tt, '{}-results-pretty.txt'.format(classifierCode.upper()))
        headerLine = 'name,resnet_ori,resnet_fgsm_adv,resnet_bim_adv,proposed_ori,proposed_fgsm_adv,proposed_bim_adv\n'
        saveLinesTo(csvStrings, '{}-results-csv.txt'.format(classifierCode.upper()), header=headerLine)

    else:
        #determine the target result
        resultFile = '{}_{}_results.txt'.format(classifierCode, datasetName)
        targetResultFile = os.path.join(resultPath, resultFile)
        fileLines = None
        try:
            with open(targetResultFile) as handle:
                fileLines = handle.readlines()

        except:
            print('Could not find result file.')
            exit()

        results = obtainResult(fileLines)
        tt, _ = updateTextTable(tt, resultFile, results, referenceResults[datasetName])
        print(tt.draw())

        
        