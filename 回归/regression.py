from numpy import *

def loadDataSet(fileName): #读取数据
    numFeat = len(open(fileName).readline().split('\t')) - 1 #第一列数据都为1表示系数b*1，第二列为x1,...最后一列为y
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0: #判断XTX的行列式是否为0
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat) #在行列式不为0的情况下，XTX存在逆矩阵，从而求出回归系数w的最优预测
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0): #输入需要预测的点的x坐标，训练用的x,y坐标，参数k
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0] #有多少个数据点
    weights = mat(eye((m))) #创建一个单位对角矩阵
    for j in range(m): #更新weights权重
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat) #在lwlr算法下XTX的计算需要考虑权重了
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws #预测出y的值

def lwlrTest(testArr,xArr,yArr,k=1.0): #若要预测多个点的y值，则需要多次遍历lwlr函数
    m = shape(testArr)[0] #计算要预测多少次
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr,yHatArr): #计算误差平方和
    return ((yArr-yHatArr)**2).sum()

#此时我们可能会遇到一个问题，如果数据的特征比样本点还多怎么办（n>m),此时矩阵不是满秩矩阵，求逆会出现问题
#对此我们引入岭回归（ridge regression)

def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat #n*n的矩阵,但是由于秩的原因一定没有逆矩阵
    denom = xTx + eye(shape(xMat)[1]) * lam #n*n的矩阵
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0) #对列求平均值
    yMat = yMat - yMean #对y进行标准化
    xMeans = mean(xMat, 0) #对每一列xi求平均值
    xVar = var(xMat, 0) #对每一列xi求方差
    xMat = (xMat - xMeans) / xVar #标准化xi
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1])) #30*n matrix用来存放30次不同lambda计算得到的回归系数
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10)) #在30个不同的lambda下调用ridgeRegres
        wMat[i, :] = ws.T
    return wMat

def regularize(xMat): #标准化X矩阵
    inMat = xMat.copy()
    inMeans = mean(inMat,0)
    inVar = var(inMat,0)
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr, yArr, eps=0.01, numIt=100): #eps为步长，numIt为外循环次数
    xMat = mat(xArr) #m*n matrix
    yMat = mat(yArr).T #1*m matrix
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt, n)) #用于储存numIt次循环得到的系数矩阵
    ws = zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy() #用于储存最优解
    for i in range(numIt): #总共进行numIt次外循环，相当于得到n次改进
        print(ws.T) #先输出上次循环结束得到的系数矩阵
        lowestError = inf
        for j in range(n): #每次外循环中，对n个回归系数都进行迭代，加上或减去步长，若总误差减小，则更新回归系数
            for sign in [-1,1]: #表示要对正负两个方向都进行测试
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE #更新最小误差
                    wsMax = wsTest #更新第j个回归系数
        ws = wsMax.copy() #第i次的内循环结束，得到系数矩阵
        returnMat[i, :] = ws.T
    return returnMat


from time import sleep
import json
import urllib.request
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    #retX（样本玩具特征矩阵），retY(样本玩具的真实价格），setNum（获取样本的数量），yr(样本玩具的年份),numPce(样本玩具的零件数），origPce(原始价格）
    sleep(10) #先休眠10秒，防止短时间内有过多的API调用
    #拼接查询的url字符串
    myAPIstr = 'get from code.google.com'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL) #利用urllib访问url地址
    retDict = json.load(pg.read()) #利用json打开和解析url获得的数据，数据信息存入字典中
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new': #判断是否为新品
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories'] #得到当前目录产品的库存列表
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5: #若价格大于原价的50%，则认为该套装完整,并将数据录入
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)

def setDataCollect(retX, retY): #用于多次调用searchForSet函数
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

def crossValidation(xArr,yArr,numVal=10): #numVal为算法中交叉验证的次数
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal,30)) #用于储存每次循环中，30组回归系数的误差
    for i in range(numVal): #总共进行numVal次外循环
        trainX = []
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList) #对indexList进行混洗，以确保之后的训练集的挑选是随机的
        for j in range(m):
            if j < m*0.9: #将90%的数据用于训练，剩余10%用作测试
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY) #保存得到30组不同lambda岭回归所得到回归系数（详情见ridgeTest函数）
        for k in range(30): #分别用30组回归系数进行测试
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)
            errorMat[i,k]=rssError(yEst.T.A,array(testY)) #存储第i次交叉验证中，第k组岭回归等得到的系数的误差
    meanErrors = mean(errorMat,0) #30次岭回归平均误差矩阵
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)] #根据最小误差找到最佳系数矩阵
    #为了将其与standRegres得出的系数矩阵作比较，将数据还原成非标准化
    xMat = mat(xArr)
    yMat=mat(yArr).T
    meanX = mean(xMat,0)
    varX = var(xMat,0)
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat))
