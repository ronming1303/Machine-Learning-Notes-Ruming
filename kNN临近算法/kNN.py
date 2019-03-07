#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:16:24 2019

@author: Ron
"""

from numpy import *
import operator
import numpy as np


def createDataSet():
    group = array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # 得到文本行数
    returnMat = np.zeros((numberOfLines, 3))  # 创建以零填充的矩阵，为了简化，另外的一个维度设为3
    classLabelVector = []  # 返回标签
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 去掉所有的回车字符
        listFromLine = line.split('\t')  # 将整行数据分割成一个元素列表
        returnMat[index, :] = listFromLine[0:3]  # 选取前三个元素存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 将数值转化为0-1标准化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(shape(dataSet))
    m = dataSet.shape[0]  # 数组的大小
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 注意事项：特征值矩阵有1000*3个值。而minVals和range的值都为1*3.为了解决这个问题使用numpy中tile函数将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide矩阵除法
    return normDataSet, ranges, minVals


# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print(("the classifier came back with: %d, the real answer is: %d") % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print(("the total error rate is: %f") % (errorCount / float(numTestVecs)))
    print(errorCount)


# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input( \
        "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per years?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: " +
          resultList[classifierResult - 1])


# 将图像转化为向量
# 把一个32*32的二进制图像转化为1*1024的向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 手写数字识别系统
# listdir是从os模块中导入函数listdir,它可以列出给定目录的文件名
from os import listdir


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)  # 得到目录中有多少文件
    # 接着，代码创建一^ 行1024列的训练矩阵，该矩阵的每行数据存储一个图像。我们可以从文件名中解析出分类数字© 。该目录下的文件按照规则命名，如文件
    # 9_45加的分类是9，它是数字9的第45个实例。然后我们可以将类代码存储在匕hwLabels向量中，使
    # 用前面讨论的img2vector函数载入图像。在下一步中, 我们对testDigits目_录中的文件执行相似的
    # 操作’ 不同之处是我们并不将这个目录下的文件载人矩阵中，而是使用classify0()函数测试该
    # 目录下的每个文件。由于文件中的值已经在0和1之间,本节并不需要使用2.2节的autoNorm()函数。
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, \
                                     trainingMat, hwLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d' \
              % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print('\nthe total number of errors is: %d' % errorCount)
    print('\nthe total error rate is: %f' % (errorCount / float(mTest)))

