from __future__ import division
import csv
import math
import argparse


def main():

    parser = argparse.ArgumentParser()

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

    args = vars(parser.parse_args())

    fp = open('trainData.csv','w')

    writer = csv.writer(fp)
    data = []

    with open(args['train_dataset']) as f:
            testData = f.read()
            testData = testData.split("\n")

            for line in testData:
                line = line.split()
                writer.writerow(line)
    fp.flush()
    fp.close()

    filePath = 'trainData.csv'

    with open(filePath) as f:
        values = f.readlines()
        for line in values:
            line = line.split(",")
            if len(line) > 0:
                data.append(line)

    # Compute the total frequency of each words and store it in a dictionary.
    # We are storing the words with its frequency separately for spam and ham
    # emails.

    spamEmails = 0
    hamEmails = 0
    totalEmails = 0
    dictSpam = {}
    dictham = {}

    for line in data:
        if len(line) > 1 and line[1] == 'spam':
            spamEmails += 1
            totalEmails += 1
            end = len(line) -2
            i = 2

            while i <= end:
                word = line[i]
                freq = line[i+1]
                dictSpam[word] = dictSpam.get(word,0) + int(freq)
                i += 2

        elif len(line) > 1 and line[1] == 'ham':
            hamEmails += 1
            totalEmails += 1
            end = len(line) - 2
            i = 2

            while i <= end:
                word = line[i]
                freq = line[i + 1]
                dictham[word] = dictham.get(word, 0) + int(freq)
                i += 2

    cleanDict(dictSpam, dictham)

    totalWordsinSpam = sum(dictSpam.values())
    totalWordsinHam = sum(dictham.values())
    totalWords = totalWordsinSpam + totalWordsinHam

    # Compute the probability of every word occurring in spam and ham separately based on their frequency
    # Laplace smoothing added for the words in the training set
    dictProbSpam = {}
    dictProbHam = {}

    for k,v in dictSpam.items():
        dictProbSpam[k] = (dictSpam.get(k)+1)/(totalWordsinSpam+1*totalWords)

    for k,v in dictham.items():
        dictProbHam[k] = (dictham.get(k)+1)/(totalWordsinHam+1*totalWords)

    # Compute the probability of spam emails and ham emails based on training data

    totalProbSpam = spamEmails/totalEmails
    totalProbHam =  hamEmails/totalEmails

    # Compute the total probability of the email being a spam and a ham based on the
    # probabilities of the words contained in the email. Instead of calculating the
    # product over all the probabilities, we have taken log and done summation over
    # the values to avoid floating values underflow.

    answerDict = dict()
    countHam = 0
    countSpam = 0

    with open(args['test_dataset']) as f:
        testData = f.readlines()
        for line in testData:
            spamProb = 0
            hamProb = 0
            if len(line) > 1:
                line = line.split()
                i = 2
                end = len(line) - 2
                while i <= end:
                    word = line[i]
                    freq = int(line[i+1])

                    if dictProbSpam.get(word, 0) > 0:
                        spamProb += math.log(dictProbSpam.get(word,0))
                    if dictProbHam.get(word,0) > 0:
                        hamProb += math.log(dictProbHam.get(word,0))
                    i += 2
                spamProb += math.log(totalProbSpam)
                hamProb += math.log(totalProbHam)

                emailId = line[0]

    # Based on the computed probability of email being a spam and ham, we compare
    # the probabilities and which ever is greater, we assign that label to the
    # test emailId

                if spamProb > hamProb:
                    answerDict[emailId] = "spam"
                else:
                    answerDict[emailId] = "ham"
                if line[1] == 'ham':
                    countHam += 1
                else:
                    countSpam += 1

    # For computing the accuracy on the test data, we have compared the assigned labels with
    # the correct labels provided. For each correct assignment, we are incrementing the correct
    # count by 1, otherwise wrong count by 1.

    t = dict()
    with open(args['test_dataset']) as f:
        data = f.readlines()
        for line in data:
            if len(line) > 1:
                line = line.split()
                t[line[0]] = line[1]
    correct = 0
    wrong = 0
    for k in t.keys():
        if t[k] == answerDict[k]:
            correct +=1
        else:
            wrong +=1

    # print str((correct/len(t))*100) + str("%")
    # print str((wrong/len(t))*100) + str("%")

    fp = open(args['output_file'],'w')
    writer = csv.writer(fp)
    ids = sorted(t.keys())
    for k in ids:
        line = [k,answerDict[k]]
        writer.writerow(line)


    # Remove the words having almost similar frequencies in both spam and ham emails
    # Having such words make for bad features. Also, remove the common articles like
    # a, an, the which does not provide any significant difference in spam and ham.
    # We are removing all such occurrences from our dictionary and these are not
    # considered for training the model

def cleanDict(dictSpam, dictham):

    for k in dictSpam.keys():
        if k in dictham:
            if abs(dictSpam[k] - dictham[k]) < 2 or k == 'and' or k == 'a' or k == 'an' or k == 'the' or k == 'of':
                del dictSpam[k]
                del dictham[k]

if __name__ == '__main__':
    main()