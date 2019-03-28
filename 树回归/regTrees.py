from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine) #放进dataMat列表中
    return dataMat

def binSplitDataSet(dataSet, feature, value): #输入的参数为数据集合matrix，待切分的特征，该特征的某个值
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:] #找到第一个feature > value的那组数据存储在mat0中
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:] #找到第一个feature <= value的那组数据存储在mat1中
    return mat0,mat1

def regLeaf(dataSet):
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: #如果满足停止条件，chooseBestSplit将返回None和某类模型的值
        return val #如果构建的是回归树，该模型是一个常数。如果是模型树，其模型是一个线性方程
    retTree = {} #如果不满足停止条件，chooseBestSplit将创建一个新的python字典并将数据集分为左右两份
    retTree['spInd'] = feat #用于分类的参数的序号
    retTree['spVal'] = val #用于分类的参数的临界值
    lSet, rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0] #容许的误差下降至
    tolN = ops[1] #切分的最小样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #如果dataSet的最后一列（分类标签）的集合只有一个值（表明是同一类），则停止继续划分
        return None, leafType(dataSet) #返回None, leafType
    m,n = shape(dataSet) #m为有多少数据，n为特征个数
    S = errType(dataSet) #计算总的误差平方和
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1): #出去最后一个分类标签，剩下的才是特征的个数
        for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]): #将第featIndex的所有出现的值放入一个集合并且遍历
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal) #分别按照splitVal进行左右分类划分
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue #如果左右两侧有一侧的数据个数少于tolN了，则不考虑此次划分（视为无效），重新进行遍历
            newS = errType(mat0) + errType(mat1) #如果此次划分有效，则记录errType
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #整个外循环结束，得到了bestIndex和对应的bestValue
    if (S - bestS) < tolS: #如果划分并没有使得误差平方和显著小于tolS，则不进行划分
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue) #如果有效，则进行划分，得到mat0, mat1
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): #防止划分过细，产生过拟合（个人认为可以删除，因为在前面已经防止划分过细了？）
        return None, leafType(dataSet)
    return bestIndex,bestValue


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)  # if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):  # if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    # if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print
            "merging"
            return treeMean
        else:
            return tree
    else:
        return tree


def isTree(obj): #输出布尔值，判断当前处理的节点是否是叶节点
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0 #如果找到两个叶节点，则计算它们的平均值（对树进行塌陷处理）


def prune(tree, testData): #输入的参数为待剪枝的树tree,剪枝所需的测试数据testData
    if shape(testData)[0] == 0:
        return getMean(tree) #没有测试数据则直接求均值对树进行塌陷处理
    if (isTree(tree['right']) or isTree(tree['left'])): #只有左右两边的树不都是叶节点，则进行划分
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal']) #得到左右两边的树lSet, rSet
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet) #十分不理解prune输出的内容为什么可以赋值到tree['left']上的,并持续迭代直到出现叶节点
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']): #如果两边都是叶节点,如果合并以后的误差减小了，则合并，否则不合并
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def linearSolve(dataSet): #将数据格式化成目标变量y和自变量x
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1] #存储属性值
    Y = dataSet[:,-1] #存储分类值
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet): #当数据不再需要切分时，生成叶节点模型，输出系数矩阵
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet): #输出误差平方和
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))


def regTreeEval(model, inDat): #用于回归树数值的计算，为了保持参数结构的一致，仍然将inDat放入参数中
    return float(model)


def modelTreeEval(model, inDat): #用于模型树数值  的计算
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval): #针对某一个单独待测的数据indata，我们尝试计算他的value
    if not isTree(tree): #如果是叶节点，直接计算value
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']: #如果数据indata大于spVal,意味着要进入左树
        if isTree(tree['left']): #若左树不是叶节点
            return treeForeCast(tree['left'], inData, modelEval)  #则继续向下划分
        else: #若是叶节点
            return modelEval(tree['left'], inData) #计算value
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval): #testData内包含多组数据，进行多次迭代得到多个value
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

