# Author: Mubarak Abdu-Aguye <maguye007.reloaded@gmail.com>
# License: BSD-3-Clause

#This script gives us stats for the degradation in performance for a given classifier
import csv
import numpy as np
import os

resultDir = "."

if __name__ == "__main__":
    #first off, enumerate the result files
    classifier = input("Enter the classifier code: ").strip().capitalize()
    resultFile = os.path.join(resultDir, "{}-results-csv.txt".format(classifier))
    bimDegradations , fgsmDegradations = [],[]

    with open(resultFile) as readHandle:
        raw = csv.DictReader(readHandle)
        for row in raw:
            if (row['proposed_ori'] == 'nan'):
                continue
            percentDegradation = (float(row['proposed_ori']) - float(row['proposed_bim_adv'])) / float(row['proposed_ori'])
            bimDegradations.append(percentDegradation)
            percentDegradation = (float(row['proposed_ori']) - float(row['proposed_fgsm_adv'])) / float(row['proposed_ori'])
            fgsmDegradations.append(percentDegradation)
    
    #now, compute the aggregated statistics
    bimDegradations = np.array(bimDegradations)
    fgsmDegradations = np.array(fgsmDegradations)

    print("BIM: ")
    print("Mean, Std: {}, {}".format(np.mean(bimDegradations), np.std(bimDegradations)))
    print("Median: {}".format(np.median(bimDegradations)))
    print("FGSM: ")
    print("Mean, Std: {}, {}".format(np.mean(fgsmDegradations), np.std(fgsmDegradations)))
    print("Median: {}".format(np.median(fgsmDegradations)))