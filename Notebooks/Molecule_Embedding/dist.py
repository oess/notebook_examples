#!/usr/bin/env python
#############################################################################
#  Copyright (C) 2015  OpenEye Scientific Software, Inc.
#############################################################################
from __future__ import print_function
from __future__ import division

import sys

try:
    import numpy as np
    import scipy as sp
    from scipy.spatial.distance import squareform
    from ipywidgets import IntProgress
    from IPython.display import display
except ImportError:
    print("Missing a required package. Please insure the following packages are installed")
    print(" numpy \n scipy")
    sys.exit(1)


from openeye.oechem import *
from openeye.oegraphsim import *


class CachingMolDistance():

    def __init__(self,dfunc):
        self.DistFunc = dfunc
        # Molecule value is order in added.  Reverse order for distance array
        self.SmilesIdx = {}
        # Molecules kept in reverse order
        self.MolIdx = []
        self.nMols = 0
        self.DistArray = np.empty(0)

    def __repr__(self):
        return("Caching Distance: Pairwaise distances of {} molecules cached".format(self.nMols))

    def dist(self, molA, molB):
        """
        Looks up distance if molecules are known, or calculates and caches if unknown
        :param molA:
        :param molB:
        """
        smiA = OEMolToSmiles(molA)
        if smiA not in self.SmilesIdx:
            self.update(molA)
            idxA = 0
        else:
            idxA = self.nMols - self.SmilesIdx[smiA] - 1

        smiB = OEMolToSmiles(molB)
        if smiB not in self.SmilesIdx:
            self.update(molB)
            idxB = 0
            idxA += 1
        else:
            idxB = self.nMols - self.SmilesIdx[smiB] - 1

        # try:
        #     assert self.DistArray[idxA,idxB] == self.DistFunc(molA,molB)
        # except:
        #     print("Distances don't match: Lookup {}  Calc {}".format(self.DistArray[idxA,idxB],self.DistFunc(molA,molB)))
        #     print(self.DistArray)
        #     print(self.SmilesIdx)
        #     print(idxA,idxB)
        #     print(smiA,smiB)

        # print(self.DistArray)
        return self.DistArray[idxA,idxB]

    def load(self, mol_iterable):
        size = len(mol_iterable)
        prog = IntProgress(min=0,max=size*(size-1)/2)
        prog.description = "Working..."
        count = 0
        display(prog)
        for m in mol_iterable:
            count += 1
            self.update(m)
            prog.value=count*(count-1)/2

    def update(self,newMol):
        if self.DistArray.shape[0] != 0:
            dVec = squareform(self.DistArray)
        else:
            dVec = np.empty(0)
        self.SmilesIdx[OEMolToSmiles(newMol)] = self.nMols
        # print("Dvec",dVec)

        newDist = np.zeros(self.nMols)


        if self.nMols > 0:
            # for i in range(1,self.nMols):
            #     print(i)
            #     newDist[-i] = self.DistFunc(newMol,self.MolIdx[-i])
            for i in range(self.nMols):
                # print(i)
                newDist[i] = self.DistFunc(newMol,self.MolIdx[-(i+1)])
        else:
            newDist = np.empty(0)

        # print("newDist",newDist)
        self.nMols += 1
        self.MolIdx.append(OEMol(newMol))
        dVec = np.concatenate((newDist,dVec))
        self.DistArray = squareform(dVec)

    def construct_dist_array(self, molIter):
        n = len(molIter)
        dv = []
        for i in range(n):
            for j in range(i+1,n):
                dv.append(self.dist(molIter[i],molIter[j]))

        return squareform(np.asarray(dv))

    def write_to_mol(self,filename="cacheDistMol.oeb.gz"):
        ofs = oemolostream()
        ofs.open(filename)
        for i in range(self.nMols):
            mol = self.MolIdx[i]
            mol.SetData("DistCache_Dists",self.DistArray[-(i+1),:])
            mol.SetData("DistCache_Idx",i)
            OEWriteMolecule(ofs,mol)

        ofs.close()

    def read_from_mol(self,filename="cacheDistMol.oeb.gz"):
        ifs = oemolistream()
        ifs.open(filename)
        self.MolIdx = []
        for mol in ifs.GetOEMols():
            self.MolIdx.append(OEMol(mol))

        self.nMols = len(self.MolIdx)

        self.DistArray = np.zeros((self.nMols,self.nMols))
        for i in range(self.nMols):
            self.DistArray[-(i+1), :] = self.MolIdx[i].GetData("DistCache_Dists")


def MolFPDist(molA, molB, fpType = OEFPType_Path):
    fpA = OEFingerPrint()
    OEMakeFP(fpA, molA, fpType)
    fpB = OEFingerPrint()
    OEMakeFP(fpB, molB, fpType)
    tan = OETanimoto(fpA, fpB)
    return 1.-tan
