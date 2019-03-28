from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1] #1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([]) #创建一个空的集合
    for document in dataSet:
        vocabSet = vocabSet | set(document) #将每篇文档返回的新词集合添加到该集合中（|表示求并集）
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet): #输入的参数为词汇表，某个文档
    returnVec = [0] * len(vocabList) #先创建一个与词汇表等长的向量，全部设置0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 #若输入的文档中的词也在词汇表中，则值变为1
        else:
            print(word + ' is not in my Vocabulary!')
    return returnVec #输出文档向量，0表示词没有出现，1表示出现过

def trainNB0(trainMatrix, trainCategory): #输入的参数为文档矩阵trainMatrix，0表示这个词在该片文档未出现，1表示出现了
    # 以及由每篇文档类别标签所构成的向量trainCategory,0表示正常，1表示侮辱性
    numTrainDocs = len(trainMatrix) #计算有多少片文档
    numWords = len(trainMatrix[0]) #计算一共有多少个不同的单词
    pAbusive = sum(trainCategory)/float(numTrainDocs) #侮辱性文档的概率=总的侮辱性文档/numTrainDocs
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0 #防止p1Vect或p0Vect为零导致累计相乘的结果为0
    for i in range(numTrainDocs): #遍历所有文档
        if trainCategory[i] == 1: #如果文档是侮辱性文档
            p1Num += trainMatrix[i] #0表示未出现，
            p1Denom += sum(trainMatrix[i]) #侮辱文档中词的总个数
        else:
            p0Num += trainMatrix[i] #0表示未出现的，
            p0Denom += sum(trainMatrix[i]) #正常文档中词的总数
    p1Vect = log(p1Num/p1Denom) #侮辱文档中各词出现的概率（相当于是先验概率p(wi|c1)）,并取Log防止下溢出
    p0Vect = log(p0Num/p0Denom) #正常文档中各词出现的概率
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1): #输入需要分类的向量vec2CLassify，以及trainNB0中计算得到的三个概率值
    p1 = sum(vec2Classify * p1Vec) + log(pClass1) #函数的含义为p(w|c1)*p(c1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1) #函数的含义为p(w|c2)*p(c2)
    if p1 > p0:
        return 1 #应归类为1
    else:
        return 0

def testingNB(): #作为一个便利函数，封装之前的所有操作与测试
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry)
    print('classified as: ' + str(classifyNB(thisDoc, p0V, p1V, pAb)))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry)
    print('classified as: ' + str(classifyNB(thisDoc, p0V, p1V, pAb)))

#目前为止，我们将每个词的出现与否作为一个特征，这样成为词集模型。如果一个词在文档中出现不止一次，
#我们应该将其考虑为词袋模型

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString): #去除文本的标点符号，空格，单词全部小写
    import re
    listOfTokens = re.split(r'\w*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]
    classList = []
    fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read()) #读取第i个垃圾邮件的所有单词
        docList.append(wordList) #将第i个wordList列表放到docList中
        fullText.extend(wordList) #将所有单词都放在fullText列表中
        classList.append(1) #classList是由25个"1"构成的列表
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList) #创建一个无重复的单词列表（包含spam和ham中所有的单词）
    trainingSet = range(50) #总共50个邮件
    testSet=[]
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet))) #从50个邮件中随机取出10个做测试
        testSet.append(trainingSet[randIndex]) #testSet=[1,5,3,6,...]表示这10个测试邮件的对应标号
        del(trainingSet[randIndex]) #把挑选出的10封邮件的标号从trainingSet中除去
    trainMat=[]
    trainClasses = []
    for docIndex in trainingSet: #对于剩下的40封邮件做训练
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) #trainMat中的每一个列表表示了每一封邮件单词出现的个数
        trainClasses.append(classList[docIndex]) #如果是1表示在spam,0表示在ham
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet: #用剩下的10封邮件进行测试
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error" + docList[docIndex])
    print('the error rate is: ' + float(errorCount)/len(testSet))