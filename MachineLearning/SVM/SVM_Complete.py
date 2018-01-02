#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: SVM_Complete.py 
@desc:
@time: 2018/01/02 
"""

from numpy import *
import matplotlib.pyplot as plt





class optStruct:
    '''
     建立的数据结构来保存所有的重要值
    '''
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        '''

        :param dataMatIn:
        :param classLabels:
        :param C:
        :param toler:
        :param kTup:
        '''
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler

        # 数据的行数
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0

        # 误差缓存，第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值
        self.eCache = mat(zeros((self.m, 2)))

        # m行m列的矩阵
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

def kernelTrans(X, A, kTup):
    '''
      转换核函数
    :param X:  dataMatIn数据集
    :param A:  dataMatIn数据集的第i行的数据
    :param kTup: 核函数的信息
    :return:
    '''
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # 径向基函数的高斯版本
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

def loadDataSet(fileName):
    """loadDataSet（对文件进行逐行解析，从而得到第行的类标签和整个数据矩阵）

    Args:
       fileName 文件名
    Returns:
        dataMat  数据矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def clacEk(oS, k):
    '''
      求 Ek误差：预测值-真实值的差
    :param oS:
    :param k:
    :return: Ek 预测结果与真实结果对比，计算误差Ek
    '''
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def innerL(i, oS):
    '''
      内循环代码
    :param i: 具体的某一行
    :param os: optStruct对象
    :return:
        0 找不到最优的值
        1 找到了最优的值，并且oS.Cache到缓存中
    '''
    # 求EK误差：预测值 - 真实值的差
    Ei = clacEk(oS, i)

    # 约束条件 (KKT条件是解决最优化问题时用到的一种方法，这里提到的最优化问题是指对于给定的某一函数，求其在指定作用于上的全局最小值)
    # 0 < alphas[i] <= C 但由于0和C是边界值，我们无法进行优化，因此需要增加一个alphas和降低一个alphas
    # 表示发生错误的概率: labelMat[i]*Ei 如果超出了toler，才需要优化，正负号，考虑绝对值



def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    '''
      完整SMO算法外循环：
    :param dataMatIn: 数据集
    :param classLavels: 类标签
    :param C: 松弛变量(常量值)， 允许有些数据点可以处于分割面的错误一侧
              控制最大化间隔和保证大部分的函数间隔 < 1.0
    :param toler: 容错率
    :param maxIter: 推出前最大的循环 次数
    :param kTup: 包含核函数的元组
    :return: b: 模型的常量值
             alphas: 拉格朗日乘子
    '''
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 循环遍历： 循环maxIter次， 并且
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0

        # 当entireSet=True or 非边界alpha对没有了，就开始寻找alpha对，然后决定是否要进行else
        if entireSet:
            # 遍历数据集上所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对
                alphaPairsChanged += innerL(i, os)



if __name__=='__main__':
    # 无核函数的测试
    # 获取特征和目标向量
    dataArr, labelArr = loadDataSet('F:\project\MachineLearning\MachineLearning\Data\SVM\\testSet.txt')
    print(labelArr)

    # b是常量值， alphas是拉格朗日乘子
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)