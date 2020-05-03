# python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/4/28 6:10
# @Author  : XjKai
# @FileName: dnn.py
# @Software: PyCharm

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
import dataProcess as dp
import time

class Model:
    """
    自定义全连接神经网络模型
    """
    def __init__(self,learn_rate=0.08, layerStructure=[3,1],dropout_rate=None, activation_func="relu", o_activation_func="softmax",
            loss_func="crossEntropy", regularization="L2", regularization_rate=0.0004, optimizer="Adam", isInitializeWeightMatrix=True):
        """
        :param learn_rate:学习率
        :param layerStructure: 网络结构，如两层结构(第一层三个神经元，第二层一个)==>[3,1]
        :param dropout_rate: dropout概率，每个隐藏层对应的dropout概率(使用dropout后，损失函数不再有意义)==>[0.2,0]
        :param activation_func: 隐藏层激活函数，可选：relu、sigmoid、tanh
        :param o_activation_func:  输出层激活函数，可选：softmax、sigmoid
        :param loss_func: 损失函数，可选：crossEntropy、mse
        :param regularization: 正则化，可选：none、L1、L2
        :param optimizer: 梯度下降优化器，可选SGD、SGDM、Adam、RMSProp、Adagrad
        :param isInitializeWeightMatrix: 是否初始化权重矩阵
        """
        self.loss = []
        self.test_accuracy = []
        self.layers_num = len(layerStructure)
        self.dropout_rate = dropout_rate if dropout_rate else [0]*self.layers_num
        self.learn_rate = learn_rate                            #hyperparameter
        self.layerStructure = layerStructure                    #hyperparameter
        self.activation_func = activation_func
        self.o_activation_func = o_activation_func
        self.loss_func = loss_func
        self.regularization = regularization
        self.regularization_rate = regularization_rate          #hyperparameter
        self.optimizer = optimizer
        self.isInitializeWeightMatrix = isInitializeWeightMatrix
        self.w = []   # 训练参数
        self.b = []   # 训练参数
        self.z = []   # z = x * w.T
        self.a = []   # a = g(z)
        self.dg = []  # dg = da/dz
        self.dz = []  # dz = dL/dz = dL/da * da/dz
        self.dw = []  # dw = dL/dw = dL/dz * dz/dw
        self.db = []  # db = dL/db = dL/dz * dz/db = dL/dz
        self.m_w, self.m_b = [], []# 一阶动量mt
        self.V_w, self.V_b = [], []# 二阶动量Vt
        self.beta = 0.9        #hyperparameter
        self.beta1 = 0.999     #hyperparameter
        nd = np.random.RandomState(14)
        for n in range(len(layerStructure)):
            if n+1 < len(layerStructure):
                self.w.append(nd.random([layerStructure[n], layerStructure[n+1]]))
            self.b.append(nd.rand(1, layerStructure[n]))

    def __activation_f(self, activation, z, dropout=0):
        """
        激活函数
        :param activation:
        :param z:
        :return: z经过激活函数后的值a及对z的导数dg
        """
        a = z
        # 根据dropout概率随机决定该层哪些神经元需要dropout
        d = np.random.rand(z.shape[0], z.shape[1]) >= dropout
        if activation.lower() == "relu":
            a[a < 0] = 0
            dg = a.copy()
            dg[dg > 0] = 1
        elif activation.lower() == "sigmoid":
            a = 1/(1+np.exp(-a))
            dg = a*(1-a)
        elif activation.lower() == "tanh":
            m = np.exp(a)
            n = np.exp(-1*a)
            a = (m-n)/(m+n)
            dg = 1 - a*a
        else: # activation.lower() == "softmax"
            a = np.exp(a)
            b = np.sum(a, axis=1, keepdims=True)
            a = a/b
            dg = a * (1 - a)
        a = a * d / (1 - dropout)
        dg = dg * d / (1 - dropout)
        return {"a": a, "dg": dg}

    def __loss_f(self, y_hat, y_true):
        """
        损失函数(使用dropout后，损失函数不再具有意义，此时的损失函数很不平稳，这是因为使用dropout后，每次计算损失函数时所对应的网结构都不同)
        :param y_hat:
        :param y_true:
        :return: 损失值
        """
        alpha = 0.000000001
        y_hat[y_hat==0] += alpha         #避免出现log0
        y_hat[y_hat==1] += (1 + alpha)   #避免出现log0
        if self.loss_func == "crossEntropy":
            loss = -1*np.sum(y_true*np.log(y_hat) + (1-y_true)*np.log(1-y_hat))/y_hat.shape[0]
            dz_hat = -(y_true - y_hat)
        elif self.loss_func == "mse":
            loss = np.sum(np.power(y_hat - y_true, 2))/(2*y_hat.shape[0])
            dz_hat = (y_hat - y_true)*self.dg[-1]

        loss_w = 0
        if self.regularization == "none":
            pass
        elif self.regularization == "L1":
            for w in self.w:
                loss_w += np.sum(np.abs(w))
        elif self.regularization == "L2":
            for w in self.w:
                loss_w += np.sum(np.power(w, 2))
        loss_w = loss_w*self.regularization_rate/y_hat.shape[0]
        loss = loss + loss_w
        return {"loss": loss, "dz_hat": dz_hat}

    def __regularization_f(self, m):
        """
        正则化
        :param m: 样本数
        :return: None
        """
        if self.regularization == "none":
            pass
        elif self.regularization == "L1":
            for i in range(len(self.dw)):
                self.dw[i] += np.abs(self.regularization_rate)/m
        elif self.regularization == "L2":
            for i in range(len(self.dw)):
                self.dw[i] += np.abs(self.regularization_rate * self.dw[i])/m

    def __initializeWeightMatrix_f(self):
        """
        初始化权重矩阵，缓解梯度爆炸/梯度消失
        (根据前一层神经元个数初始化)
        :return: None
        """
        if self.isInitializeWeightMatrix:
            if self.activation_func == "relu":
                for w in self.w:
                    w *= np.sqrt(2 / w.shape[0])
            if self.activation_func == "tanh":
                for w in self.w:
                    w *= np.sqrt(1 / w.shape[0])

    def __optimizer_f(self, learn_rate, optimizer, index,  global_step=None):
        """
        梯度下降优化器
        :param optimizer: 优化器
        :param global_step: 全局迭代数
        :return: 优化后的梯度
        """
        alpha = 0.000000001
        if optimizer == "SGDM":
            self.m_w[index] = self.beta * self.m_w[index] + (1 - self.beta) * self.dw[index]
            self.m_b[index] = self.beta * self.m_b[index] + (1 - self.beta) * self.db[index]
            # self.V_w[index] = 1
            # self.V_b[index] = 1
            m_w, m_b, V_w, V_b = self.m_w[index], self.m_b[index], self.V_w[index], self.V_b[index]
        elif optimizer == "RMSProp":
            self.m_w[index] = self.dw[index]
            self.m_b[index] = self.db[index]
            self.V_w[index] = self.beta * self.V_w[index] + (1 - self.beta) * np.square(self.dw[index])
            self.V_b[index] = self.beta * self.V_b[index] + (1 - self.beta) * np.square(self.db[index])
            m_w, m_b, V_w, V_b = self.m_w[index], self.m_b[index], self.V_w[index], self.V_b[index]
        elif optimizer == "Adagrad":
            self.m_w[index] = self.dw[index]
            self.m_b[index] = self.db[index]
            self.V_w[index] += np.square(self.dw[index])
            self.V_b[index] += np.square(self.db[index])
            m_w, m_b, V_w, V_b = self.m_w[index], self.m_b[index], self.V_w[index], self.V_b[index]
        elif optimizer == "Adam":
            self.m_w[index] = self.beta * self.m_w[index] + (1 - self.beta) * self.dw[index]
            self.m_b[index] = self.beta * self.m_b[index] + (1 - self.beta) * self.db[index]
            self.V_w[index] = self.beta1 * self.V_w[index] + (1 - self.beta1) * np.square(self.dw[index])
            self.V_b[index] = self.beta1 * self.V_b[index] + (1 - self.beta1) * np.square(self.db[index])
            m_w, m_b, V_w, V_b = self.m_w[index], self.m_b[index], self.V_w[index], self.V_b[index]
            # 修正后的一阶和二阶动量
            m_w = m_w / ((1 - np.power(self.beta, int(global_step))) + alpha)
            m_b = m_b / ((1 - np.power(self.beta, int(global_step))) + alpha)
            V_w = V_w / ((1 - np.power(self.beta1, int(global_step))) + alpha)
            V_b = V_b / ((1 - np.power(self.beta1, int(global_step))) + alpha)
        else:  # SGD
            self.m_w[index] = self.dw[index]
            self.m_b[index] = self.db[index]
            # self.V_w[index] = 1
            # self.V_b[index] = 1
            m_w, m_b, V_w, V_b = self.m_w[index], self.m_b[index], self.V_w[index], self.V_b[index]
        # 下降梯度yita = lr * mt / sqrt(Vt)
        yita_w = learn_rate * (m_w / np.sqrt(V_w))
        yita_b = learn_rate * (m_b / np.sqrt(V_b))
        return yita_w, yita_b

    def __learning_rate_decay(self, epoch_num):
        """
        学习率衰减
        :param epoch_num:
        :return:
        """
        decay_rate = 0.6  #衰减系数
        return (1 / (1 + decay_rate * epoch_num)) * self.learn_rate

    def __grad_check_f(self, x, y):
        pass

    def forward(self, test_x, test_y, w=None, b=None):
        """
        用训练好的模型前向传播
        :param test_x: 测试集x
        :param test_y: 测试集y
        :return: 预测值y_hat
        """
        if not w: w = self.w
        if not b: b = self.b
        a = []  # a = g(z)
        z = np.dot(test_x, w[0]) + b[0]
        aa = self.__activation_f(activation=self.activation_func, z=z)
        a.append(aa["a"])
        for l in range(1, self.layers_num):
            if l < self.layers_num - 1:
                activation = self.activation_func
            # 输出层
            elif l == self.layers_num - 1:
                activation = self.o_activation_func
            z = np.dot(a[l - 1], w[l]) + b[l]
            aa = self.__activation_f(activation=activation, z=z)
            a.append(aa["a"])
        return a[-1]

    def test_validation(self, test_x, test_y):
        """
        测试集验证
        :param test_x:
        :param test_y:
        :return:
        """
        y_predict = self.forward(test_x=test_x, test_y=test_y)
        rs = []
        for i in range(y_predict.shape[0]):
            rs.append(np.argmax(y_predict[i]))
        rs = np.asarray(rs).reshape(len(rs), 1)
        return sum(test_y == rs) / len(test_y)

    def fit(self, train_x, train_y, test_x, test_y, batch_size=None, epochs=1):
        """
        训练函数
        :param train_x: 训练集x
        :param train_y: 训练集y
        :param test_x: 测试集x
        :param test_y: 测试集y
        :param epochs: 迭代次数
        :param batch_size: 一个batch内样本量
        :return: None
        """
        m_batch = train_x.shape[0]  #样本数
        nd = np.random.RandomState(14)
        self.w.insert(0,nd.rand(train_x.shape[1], self.layerStructure[0])*0.01)  # 添加第一层网络的训练参数w
        self.__initializeWeightMatrix_f()       #初始化权重矩阵
        self.m_w, self.m_b = [np.zeros(m_w.shape) for m_w in self.w], [np.zeros(m_b.shape) for m_b in self.b]# 初始化一阶动量mt
        self.V_w, self.V_b = [np.ones(v_w.shape) for v_w in self.w], [np.ones(v_b.shape) for v_b in self.b]# 初始化二阶动量Vt
        global_step = 0

        print("TrainSample : {}  TestSample : {}".format(train_x.shape[0], test_x.shape[0]))

        for epoch in range(epochs):      # 训练轮数
            time_s = time.time()
            batch_count = 0
            front = 0
            rear = 0
            batch_size = batch_size if  batch_size != None else m_batch
            while True:
                if rear == m_batch: break
                front = batch_count * batch_size
                rear = (batch_count + 1) * batch_size
                rear = rear if rear <= m_batch else m_batch
                m_mini_batch = rear - front
                batch_count += 1
                global_step += 1

                tx = train_x[front:rear]
                ty = train_y[front:rear]

                self.z = []                 # z = x * w.T
                self.a = []                 # a = g(z)
                self.dg = []                # dg = da/dz
                self.dz = []                # dz = dL/dz = dL/da * da/dz
                self.dw = []                # dw = dL/dw = dL/dz * dz/dw
                self.db = []                # db = dL/db = dL/dz * dz/db = dL/dz

                # ###前向传播###
                z = np.dot(tx, self.w[0]) + self.b[0]
                aa = self.__activation_f(activation=self.activation_func, z=z, dropout=self.dropout_rate[0])
                self.z.append(z)
                self.a.append(aa["a"])
                self.dg.append(aa["dg"])
                for l in range(1,self.layers_num):
                    if l < self.layers_num - 1:
                        activation = self.activation_func
                        dropout = self.dropout_rate[l]
                    # 输出层
                    elif l == self.layers_num - 1:
                        activation = self.o_activation_func
                        dropout = 0
                    z = np.dot(self.a[l-1], self.w[l]) + self.b[l]
                    aa = self.__activation_f(activation=activation, z=z, dropout=dropout)
                    self.z.append(z)
                    self.a.append(aa["a"])
                    self.dg.append(aa["dg"])

                # ###反向传播###
                loss = self.__loss_f(y_hat=self.a[-1],y_true=ty)
                self.dz.insert(0, loss["dz_hat"])
                self.dw.insert(0, np.dot(self.a[-2].T, self.dz[0]) / m_mini_batch)
                self.db.insert(0, np.sum(self.dz[0], axis=0, keepdims=True) / m_mini_batch)
                for b in reversed(range(0, self.layers_num-1)):
                    self.dz.insert(0, np.dot(self.dz[0], self.w[b+1].T)*self.dg[b])
                    if b > 0:
                        self.dw.insert(0, np.dot(self.a[b-1].T, self.dz[0]) / m_mini_batch)
                    elif b == 0:
                        self.dw.insert(0, np.dot(tx.T, self.dz[0]) / m_mini_batch)
                    self.db.insert(0, np.sum(self.dz[0], axis=0, keepdims=True) / m_mini_batch)

                # ###参数更新###
                for i in range(len(self.dw)):
                    self.__regularization_f(m_mini_batch) #正则化
                    yita_w, yita_b = self.__optimizer_f(learn_rate=self.__learning_rate_decay(epoch), optimizer=self.optimizer, index=i, global_step=global_step)
                    self.w[i] -= yita_w
                    self.b[i] -= yita_b

            time_u = time.time() - time_s
            test_accuracy = self.test_validation(test_x, test_y)
            self.loss.append(loss["loss"])
            self.test_accuracy.append(test_accuracy)
            print("epoch_num: {:0>3d}   train_loss: {:0<.5f}   test_accuracy: {:0<.5f}   use_time: {:0<.5f}min".format(epoch, self.loss[-1], test_accuracy[0], time_u/60))


if __name__ == '__main__':
    # mnist数据集
    (x_train, y_train), (x_test, y_test) =tf.keras.datasets.mnist.load_data()# tf.keras.datasets.cifar10.load_data()
    x_train, y_train = dp.dataP(x_train, y_train, oneHot_num=10)
    x_test, y_test = dp.dataP(x_test, y_test)

    # 鸢尾花数据集
    # iris = datasets.load_iris()
    # x = iris.data
    # y = iris.target
    # x = x / (np.max(x) - np.min(x)) * 0.99 + 0.01
    # x_train, x_test, y_train, y_test = train_test_split(x, y)
    # x_train, y_train = dp.dataP(x_train, y_train, oneHot_num=3)
    # x_test, y_test = dp.dataP(x_test, y_test)

    # 训练模型
    m = Model(layerStructure=[128, 10])
    m.fit(train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test, batch_size=32, epochs=10)

    # 用训练的模型进行预测
    # y_predict = m.forward(test_x=x_test, test_y=y_test)
    # rs = []
    # for i in range(y_predict.shape[0]):
    #     rs.append(np.argmax(y_predict[i]))
    # rs = np.asarray(rs).reshape(len(rs),1)
    # print("真实值:", y_test)
    # print("预测值:", rs)
    # print("准确率:", sum(y_test == rs) / len(y_test))

    # 绘出loss曲线
    plt.title("train_loss")
    plt.xlabel("epoch_num")
    plt.ylabel("loss")
    plt.plot(m.loss, label= "$Loss$")
    plt.legend()
    # 绘出accuracy曲线
    plt.title("test_accuracy")
    plt.xlabel("epoch_num")
    plt.ylabel("accuracy")
    plt.plot(m.test_accuracy, label= "Accuracy")
    plt.legend()
    plt.show()