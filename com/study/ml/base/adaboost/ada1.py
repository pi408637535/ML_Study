# coding=gbk
from numpy import *
def loadSimpData():
    datMat=matrix([[1.0,2.1],
                   [2.0,1.1],
                   [1.3,1.0],
                   [1.0,1.0],
                   [2.0,1.0]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

#从文件加载数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#简单的单层决策分类器
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#查找数据集上的最佳的单层决策树
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #初始化总的错误数
    for i in range(n):#循环遍历所有的组合
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#循环遍岭当前的的这个矩阵
            for inequal in ['lt', 'gt']: #遍历小于或者大于的
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#用单层决策分类树进行分类
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                print("predictedVals",predictedVals.T,"errArr",errArr.T )
                weightedError = D.T*errArr  #计算出权重
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst
    
#基于单层决策树的Adaboost的训练过程
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #一开始的权重，都相等
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#查找最佳单层决策树
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#计算alpha的值
        bestStump['alpha'] = alpha  
        print("alpha",alpha)
        weakClassArr.append(bestStump)#存储最佳单层决策树
        print("classEst",classEst)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #重新计算D的值
        D = multiply(D,exp(expon)) #Calc New D, element-wise 
        D = D/D.sum()
        print("D",D)
        #计算训练错误率calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        print("aggClassEst",aggClassEst)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        #print aggErrors
        errorRate = aggErrors.sum()/m
        print(errorRate)
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

#adaboost分类器
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'], classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)


 #绘制ROC曲线，计算AUC
def plotROC(predStrengths,classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)

if __name__ == '__main__':
    
