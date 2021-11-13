#Helper code for reading UCR TS Archive 2015 files

# Author: Mubarak Abdu-Aguye <maguye007.reloaded@gmail.com>
# License: BSD-3-Clause

import numpy as np

class UCRTSLoader(object):
    def __init__(self, filename, pytorch_compatible=False):
        super(UCRTSLoader).__init__()

        self.fn = filename
        self.data = np.loadtxt(self.fn, delimiter=",")
        self.class_labels = sorted(set(self.data[:,0].astype(int)))
        self.n_classes = len(self.class_labels)
        self.class_map = {self.class_labels[n]: n for n in range(self.n_classes)}

        self.pytorch_compatible = pytorch_compatible
    
    def __getitem__(self, index):
        datum = self.data[index]
        label = int(datum[0])
        data = datum[1:]

        if not self.pytorch_compatible:
            #inject custom logic here
            data = data[:,np.newaxis]
        else:
            data = data[np.newaxis,:]
        
        return (data, self.class_map[label])

    def __len__(self):
        return len(self.data)