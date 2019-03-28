from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #转换成浮点数
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB): #两向量之间的欧氏距离
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k): #生成初始的k个随机质心
    n = shape(dataSet)[1] #数据的维度是n
    centroids = mat(zeros((k,n))) #用于存储质心的矩阵
    for j in range(n):
        minJ = min(dataSet[:,j]) #找到第j维的最小值
        rangeJ = float(max(dataSet[:,j]) - minJ) #找到j维的range,防止质心跑到数据的边界以外
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0] #计算已有数据的个数m
    clusterAssment = mat(zeros((m,2))) #clusterAssment第一列用于存储每个点的分类结果，第二列存储误差（点到质心的距离平方）
    centroids = createCent(dataSet, k) #创建k个初始质心
    clusterChanged = True
    while clusterChanged == True: #只要聚类发生了改变
        clusterChanged = False
        for i in range(m): #分别对m个数据进行聚类
            minDist = inf
            minIndex = -1
            for j in range(k): #分别计算这些数据到k个质心的距离，并更新离最短距离及相应的质心index
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2 #将分类结果，点到对应分类中心的距离平方放入assment
        print(centroids) #输出k个质心
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] #ptsInCLust用于储存所有在cent类的数据
            centroids[cent,:] = mean(ptsInClust, axis=0) #更新质心,axis=0表示沿矩阵的列方向进行均值计算
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud): #二分K-均值聚类算法
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0] #初始的1个聚类中心为所有数据的平均重心
    centList =[centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k): #只要centList里面的质心数量还少于k,就需要继续二分
        lowestSSE = inf
        for i in range(len(centList)): #遍历所有已经存在的质心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] #将所有已经分在i族里面的数据放在ptsInCurrCluster中
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) #将i族里面的数据二分类
            sseSplit = sum(splitClustAss[:,1]) #得到进行二分后那部分数据的sse
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) #将其他没有二分的数据求sse
            print("sseSplit, and notSplit: " + str(sseSplit) + " " + str(sseNotSplit))
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #把分类值换成3，4，5...
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit #此部分分类值依旧保留不改变
        print('the bestCentToSplit is: ' + str(bestCentToSplit))
        print('the len of bestClustAss is: ' + str(len(bestClustAss)))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] #将需要进行拆分的数据旧质心替换成第一个新质心
        centList.append(bestNewCents[1,:].tolist()[0]) #添加第二个新质心
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss #将更新过的数据同时更新到clusterAssment
    return mat(centList), clusterAssment


import urllib.request
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.request.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print(yahooApi)
    c=urllib.request.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print("error fetching")
        sleep(1)
    fw.close()


def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()