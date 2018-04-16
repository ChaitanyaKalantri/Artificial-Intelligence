#To accpet the inputs from the user
import argparse, os, sys
import pandas as pd
import numpy as np
from scipy.stats import chisquare

parser = argparse.ArgumentParser()

parser.add_argument('-p',
                    help='specify p-value threshold',
                    dest='pValue',
                    action='store',
                    default='0.005'
                    )

parser.add_argument('-f1',
                    help='specify training dataset path',
                    dest='train_dataset',
                    action='store',
                    default=''
                    )

parser.add_argument('-f2',
                    help='specify test dataset path',
                    dest='test_dataset',
                    action='store',
                    default=''
                    )

parser.add_argument('-o',
                    help='specify output file',
                    dest='output_file',
                    action='store',
                    default=''
                    )

parser.add_argument('-t',
                    help='specify decision tree',
                    dest='decision_tree',
                    action='store',
                    default=''
                    )


args = parser.parse_args()

#Even thought I am hard coding the input file. But, it will accept the one given on the command line 
examples = pd.read_csv('/Users/chaitanyakalantri/Desktop/AI/Project5/train.csv', header=None, sep=" ")


target_attribute = pd.read_csv('/Users/chaitanyakalantri/Desktop/AI/Project5/train_label.csv',header=None)

#Extracted the relevant attributes
examples['output'] = target_attribute[0]

features = examples.shape[1] - 1
attributes = [i for i in range(features)]

#Creating a tree node
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data

#Implementation of the entropy of the given attribute 
#For this I have implemented the entropy formula
def entropyFinder(attrib, examples):
    
    n = examples[attrib].count()
    uniqueValues = examples[attrib].unique()
    
    entropySum = 0
    
    for value in uniqueValues:
        count = (examples[attrib] == value).sum()
        p = float(count)/n
        #Create a temporary dataframe of attribute and output columns:
        tempdf = examples.filter([attrib,'output'],axis=1)
        tempdf = tempdf.loc[(tempdf[attrib]==value)]
        tempdfLen = tempdf['output'].count()
        zeroCnt = (tempdf['output'] == 0).sum()
        oneCnt = (tempdf['output'] == 1).sum()
        pZero = float(zeroCnt)/tempdfLen
        pOne = float(oneCnt)/tempdfLen
        
        if pZero == 0:
            entropyZero = 0
        else:
            entropyZero = pZero*(np.log2(pZero))
        
        if pOne == 0:
            entropyOne = 0
        else:
            entropyOne = pOne*(np.log2(pOne))
        
        tEntropy = -(entropyZero + entropyOne )
        
        entropySum += p*float(tEntropy)
    
    return float(entropySum)

#This function finds the best Attribute of all the given 274 attributes
#For finding the best attribute we are calling the entropyFinder method and whichever attribute has the 
#highest entropy will be selected as the best Attribute
def bestAttribute(examples, attrib):
    #default Values
    minEntropy = 9999999999
    bestAttribute = attrib[0]
    
    for attribute in attrib:
        if entropyFinder(attribute, examples) < minEntropy:
            bestAttribute = attribute
            minEntropy = entropyFinder(attribute, examples)
    
    return bestAttribute

#Implemented the Chisquare formula
def ChiSquare(examples, bestAttribute, pValue):
    
    fObs = list()
    fExp = list()
    
    zeroCnt = (examples['output'] == 0).sum()
    oneCnt = (examples['output'] == 1).sum()
    n = examples['output'].count()
    
    zeroRatio = float(zeroCnt)/n
    oneRatio = float(oneCnt)/n
    
    unique = examples[bestAttribute].unique()
    
    for value in unique:
        
        tempdf = examples.filter([bestAttribute,'output'],axis=1)
        tempdf = tempdf.loc[(tempdf[bestAttribute]==value)]
        valueCount = tempdf['output'].count()
        
        obsZeroes = float((tempdf['output'] == 0).sum())
        obsOnes = float((tempdf['output'] == 1).sum())
        
        expZeroes = float(zeroRatio)*valueCount
        expOnes = float(oneRatio)*valueCount
        
        #not sure check for divide by zero
        fObs.append(obsZeroes)
        fObs.append(obsOnes)
        fExp.append(expZeroes)
        fExp.append(expOnes)
    
    chiSq, p = chisquare(fObs, fExp)
    
    if p<=pValue:
        return True
    else:
        return False

#Implemeted the ID3 formula as per the wikipedia page
def ID3(examples, attributes, pValue):
    
    #if depth == 7:
    #    return

    #Check if all examples are positive:
    if (examples['output'] == 1).sum() == examples['output'].count():
        root = TreeNode('T',children=[-1]*5)
        return root
    
    #Check if all examples are negative:
    if (examples['output'] == 0).sum() == examples['output'].count():
        root = TreeNode('F',children=[-1]*5)
        return root
    
    #If attributes is empty, then select the majority element as output value
    if len(attributes)==0:
        oneCnt = 0
        zeroCnt = 0
        oneCnt = (examples['output'] == 1).sum()
        zeroCnt = (examples['output'] == 0).sum()
        if oneCnt >= zeroCnt:
            root = TreeNode('T',children=[-1]*5)
            return root
        else:
            root = TreeNode('F',children=[-1]*5)
            return root
    
    #Choose the best attribute
    A = bestAttribute(examples, attributes)
    
    if ChiSquare(examples, A, pValue):
        root = TreeNode(A, children=[-1]*5)
    
        #check if 'A' or A works
        unique = examples[A].unique()
        attributes.remove(A)

        i=0
        for value in unique:
            examplesSubset = examples.loc[examples[A] == value]

            if examplesSubset.empty:
                oneCnt = (examples['output'] == 1).sum()
                zeroCnt = (examples['output'] == 0).sum()
                if oneCnt>= zeroCnt:
                    root.nodes[i]= TreeNode('T',children[-1]*5)
                else:
                    root.nodes[i] = TreeNode('F', children[-1]*5)
            else:    
                root.nodes[i] = ID3(examplesSubset, attributes, pValue)   
            i+=1
    else:
        #check if a dummy node should be returned or "None" or something else
        return
        
    return root    

#Using BFS I will print the complete tree
def BFS(root):
    
    queue = list()
    
    queue.append(root)
    
    while len(queue)>0:
        n = len(queue)
        r = list()
        for i in range(n):
            node = queue[0]
            queue.remove(node)
            if node is not None:
                r.append(node.data)
                for children in node.nodes:
                    if children!=-1:
                        queue.append(children)
        
        print (r)


#I was copying the file because I was trying to execute the data for all the p-values.
#However, the code can be comented as we are only passing a single p-value from command line argument
from copy import deepcopy
examplesBackup = examples.copy(deep=True)
attributesBackup = deepcopy(attributes)


print("Output for 0.01 p-value\n")
examples = examplesBackup
attributes = attributesBackup

pValue = 0.01
root = ID3(examples, attributes, pValue)


BFS(root)


# print("Output for 0.05 p-value\n")
# examplesBackup = examples
# attributesBackup = attributes

# pValue = 0.05
# root = ID3(examplesBackup, attributesBackup, pValue)
# BFS(root)

# print("Output for 1 p-value\n")
# examplesBackup = examples
# attributesBackup = attributes

# pValue = 1
# root = ID3(examplesBackup, attributesBackup, pValue)
# BFS(root)
# print("Done")
