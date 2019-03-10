from numpy import *

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split() #去除空格，并拆分数据
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #[[x0=1.0, x1, x2],...]
        labelMat.append(int(lineArr[2])) #储存各个数据的分类数字
    return dataMat, labelMat

def sigmoid(inX): #构造sigmoid函数
    return 1.0/(1+exp(-inX))

#Logistic回归梯度上升优化算法
def gradAscent(dataMatIn, classLabels): #dataMathIn是loadDataSet中的dataMat
    dataMatrix = mat(dataMatIn) #将dataMathIn转换成100*3的numpy matrix
    labelMat = mat(classLabels).transpose() #将dataMathIn转换成100*1的numpy matrix
    m,n = shape(dataMatrix) #m为行数，n为列数
    alpha = 0.001 #alpha是向目标移动的步长
    maxCycles = 500 #迭代次数
    weights = ones((n,1)) #构造回归系数列向量[1,1,1](每个归回系数初始化为1)
    for k in range(maxCycles): #开始循环迭代
        h = sigmoid(dataMatrix*weights) #dataMatrix是一个100*1的列向量，取值在(0,1)之间
        error = (labelMat - h) #算得的sigmoid函数值与实际的labelMat的误差
        weights = weights + alpha * dataMatrix.transpose()* error #迭代调整回归系数
    return weights

def plotBestFit(weights): #输入的参数weights是回归系数list
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat) #一个100*3的矩阵
    n = shape(dataArr)[0] #n=100
    xcord1 = [] #存储分类1的X1
    ycord1 = [] #存储分类1的X2
    xcord2 = [] #存储分类0的X1
    ycord2 = [] #存储分类0的X2
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s') #红色方块表示分类1
    ax.scatter(xcord2, ycord2, s=30, c='green') #绿色表示分类0
    x = arange(-3.0, 3.0, 0.1) #画一条分割线，在这条分割线上，w0*x0+w1*x1+w2*x2=0(注意w0=1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#随机梯度上升算法
def stoGradAscent0(dataMatrix, classLabels): #dataMatrix=[[x0=1.0, x1, x2],...]
    m, n = shape(dataMatrix) #m为样本数量，本例中，m=100,n=3
    alpha = 0.01 #步长为0.01
    weights = ones(n) #系数初始化为1
    for i in range(m): #迭代m次
        h = sigmoid(sum(dataMatrix[i] * weights)) #每次只用一组数据进行回归系数调整，如果每次都用全部数据进行系数调整可能导致效率过低
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#改进随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n) #系数初始化为1
    for j in range(numIter): #迭代numIter次
        dataIndex = range(m)
        for i in range(m): #numIter次迭代中，每次再进行m次的随机梯度上升迭代
            alpha = 4/(1.0+j+i)+0.0001 #步长会随着迭代次数的增加逐渐变小但不会变为0，这是为了缓解多次迭代后数据的高频波动
            randIndex = int(random.uniform(0,len(dataIndex))) #随机取一个数据迭代
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(list(dataIndex)[randIndex]) #删掉用过的那组数据对应的list索引
    return weights

def classifyVector(inX, weights): #根据sigmoid函数判断分类
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt');
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines(): #分别读取每一个样本
        currLine = line.strip().split('\t') #将21个特征指标数据和1个分类标签数据存储在list中
        lineArr = []
        for i in range(21): #有21个特征指标
            lineArr.append(float(currLine[i])) #将指标数据存入lineArr
        trainingSet.append(lineArr) #将第i个指标数据列表放入trainingSet
        trainingLabels.append(float(currLine[21])) #将第i个分类标签数据放在trainingLabels中
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000) #算得回归系数matrix
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines(): #开始进行算法测试
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is " + errorRate)
    return errorRate #得到错误率

def multiTest(): #计算多次测试得到的平均错误率
    numTests = 10
    errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after" + numTests + "iterations the average error rate is"
          + errorSum/float(numTests))