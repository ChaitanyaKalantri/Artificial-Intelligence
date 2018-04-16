# To accept the inputs from the user
import argparse, os, sys

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

import pandas as pd
import numpy as np
from scipy.stats import chisquare

examples = pd.read_csv('/Users/chaitanyakalantri/Desktop/AI/Project5/train.csv', header=None, sep=" ")

target_attribute = pd.read_csv('/Users/chaitanyakalantri/Desktop/AI/Project5/train_label.csv', header=None)

examples['output'] = target_attribute[0]

features = examples.shape[1] - 1
attributes = [i for i in range(features)]


class TreeNode():
    def __init__(self, data='T', children=[-1] * 5):
        self.nodes = list(children)
        self.data = data


def entropy(attribute, examples):
    n = examples[attribute].count()
    uniqueValues = examples[attribute].unique()

    entropySum = 0

    for value in uniqueValues:
        count = (examples[attribute] == value).sum()
        p = float(count) / n
        # Create a temporary dataframe of attribute and output columns:
        tempdf = examples.filter([attribute, 'output'], axis=1)
        tempdf = tempdf.loc[(tempdf[attribute] == value)]
        tempdfLen = tempdf['output'].count()
        zeroCount = (tempdf['output'] == 0).sum()
        oneCount = (tempdf['output'] == 1).sum()
        pZero = float(zeroCount) / tempdfLen
        pOne = float(oneCount) / tempdfLen

        if pZero == 0:
            entropyZero = 0
        else:
            entropyZero = pZero * (np.log2(pZero))

        if pOne == 0:
            entropyOne = 0
        else:
            entropyOne = pOne * (np.log2(pOne))

        tEntropy = -(entropyZero + entropyOne)

        entropySum += p * float(tEntropy)

    return float(entropySum)


def chooseBestAttribute(examples, attributes):
    # default Values
    minEntropy = 9999999999
    bestAttribute = attributes[0]

    for attribute in attributes:
        if entropy(attribute, examples) < minEntropy:
            bestAttribute = attribute
            minEntropy = entropy(attribute, examples)

    return bestAttribute


def checkChiSquare(examples, bestAttribute, pValue):
    fObs = list()
    fExp = list()

    zeroCount = (examples['output'] == 0).sum()
    oneCount = (examples['output'] == 1).sum()
    n = examples['output'].count()

    zeroRatio = float(zeroCount) / n
    oneRatio = float(oneCount) / n

    uniqueValues = examples[bestAttribute].unique()

    for value in uniqueValues:
        tempdf = examples.filter([bestAttribute, 'output'], axis=1)
        tempdf = tempdf.loc[(tempdf[bestAttribute] == value)]
        valueCount = tempdf['output'].count()

        observedZeroes = float((tempdf['output'] == 0).sum())
        observedOnes = float((tempdf['output'] == 1).sum())

        expectedZeroes = float(zeroRatio) * valueCount
        expectedOnes = float(oneRatio) * valueCount

        # not sure check for divide by zero
        fObs.append(observedZeroes)
        fObs.append(observedOnes)
        fExp.append(expectedZeroes)
        fExp.append(expectedOnes)

    chiSq, p = chisquare(fObs, fExp)

    if p <= pValue:
        return True
    else:
        return False


def ID3(examples, attributes, pValue):
    # if depth == 7:
    #    return

    # Check if all examples are positive:
    if (examples['output'] == 1).sum() == examples['output'].count():
        root = TreeNode('T', children=[-1] * 5)
        return root

    # Check if all examples are negative:
    if (examples['output'] == 0).sum() == examples['output'].count():
        root = TreeNode('F', children=[-1] * 5)
        return root

    # If attributes is empty, then select the majority element as output value
    if len(attributes) == 0:
        oneCount = 0
        zeroCount = 0
        oneCount = (examples['output'] == 1).sum()
        zeroCount = (examples['output'] == 0).sum()
        if oneCount >= zeroCount:
            root = TreeNode('T', children=[-1] * 5)
            return root
        else:
            root = TreeNode('F', children=[-1] * 5)
            return root

    # Choose the best attribute
    A = chooseBestAttribute(examples, attributes)

    if checkChiSquare(examples, A, pValue):
        root = TreeNode(A, children=[-1] * 5)

        # check if 'A' or A works
        uniqueValues = examples[A].unique()
        attributes.remove(A)

        i = 0
        for value in uniqueValues:
            examplesSubset = examples.loc[examples[A] == value]

            if examplesSubset.empty:
                oneCount = (examples['output'] == 1).sum()
                zeroCount = (examples['output'] == 0).sum()
                if oneCount >= zeroCount:
                    root.nodes[i] = TreeNode('T', children[-1] * 5)
                else:
                    root.nodes[i] = TreeNode('F', children[-1] * 5)
            else:
                root.nodes[i] = ID3(examplesSubset, attributes, pValue)
            i += 1
    else:
        # check if a dummy node should be returned or "None" or something else
        return

    return root


def BFS(root):
    queue = list()

    queue.append(root)

    while len(queue) > 0:
        n = len(queue)
        r = list()
        for i in range(n):
            node = queue[0]
            queue.remove(node)
            if node is not None:
                r.append(node.data)
                for children in node.nodes:
                    if children != -1:
                        queue.append(children)

        print (r)


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





