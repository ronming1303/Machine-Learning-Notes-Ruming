from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 对currentLabel进行计数
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:]) # 将featVec中的axis=value的元素除去了
            retDataSet.append(reducedFeatVec) #将各个除去了axis元素的列表放进一个新的列表中
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) #将第i个features组成一个集合
        newEntropy = 0.0
        for value in uniqueVals: #计算如果按照第i个feature进行分类得到的香农熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature #得到能获得最佳香农熵的feture是第几个

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] #得到出现次数最多的分类名称

def createTree(dataSet, labels): 
    classList = [example[-1] for example in dataSet] #包含了数据集的所有类标签
    if classList.count(classList[0]) == len(classList): #递归函数的第一个终止条件是所有类标签完全相同
        return classList[0] #直接返回该类标签
    if len(dataSet[0]) == 1: #递归的第二个条件是用完了所有特征仍不能将数据集划分成仅包含唯一类别的分组，数据集只剩下一个类标签
        return majorityCnt(classList) #挑选出现次数最多的类别作为返回值
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:] #拷贝所有剩下的标签值
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
                                                  subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0] #先取出一个属性
    secondDict = inputTree[firstStr] #这个属性所对应的value
    featIndex = featLabels.index(firstStr) #这个属性在featLabels中对应的数字索引,这些步骤类似于将inputTree标准化成testVec一样的label顺序
    for key in secondDict.keys():
        if testVec[featIndex] == key: #读取第一个属性的对应特征值（0或1）
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec) #如果是字典，则读取第二个甚至第三个属性的对应特征值
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)