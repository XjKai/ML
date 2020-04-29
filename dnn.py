# python3.7
# -*- coding: utf-8 -*-
# @Author  : XjKai


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

class model():
    """
    全连接神经网络模型
    """
    def __init__(self,learn_rate=0.05, layerStructure=(3,1), activation_func="relu", o_activation_func="softmax",
            loss_func="crossEntropy", regularization="L2", regularization_rate=0.03, optimizer="SGD"):
        """
        :param learn_rate:学习率
        :param layerStructure: 网络结构，如两层结构(第一层三个神经元，第二层一个)==>[3,1]
        :param activation_func: 隐藏层激活函数，可选：relu、sigmoid、tanh
        :param o_activation_func:  输出层激活函数，可选：softmax、sigmoid
        :param loss_func: 损失函数，可选：crossEntropy、mse
        :param regularization: 正则化，可选：none、L1、L2
        :param optimizer: 梯度下降优化器，可选SGD、SGDM、Adam、RMSProp、Adagrad
        """
        self.loss = []
        self.layers_num = len(layerStructure)
        self.learn_rate = learn_rate
        self.layerStructure = layerStructure
        self.activation_func = activation_func
        self.o_activation_func = o_activation_func
        self.loss_func = loss_func
        self.regularization = regularization
        self.regularization_rate = regularization_rate
        self.optimizer = optimizer
        #训练参数
        self.w = []
        self.b = []
        nd = np.random.RandomState(14)
        for n in range(len(layerStructure)):
            if n+1 < len(layerStructure):
                self.w.append(nd.random([layerStructure[n], layerStructure[n+1]]))
            self.b.append(nd.random([1, layerStructure[n]]))

    def __activation_f(self,activation,z):
        """
        激活函数
        :param activation:
        :param z:
        :return:
        """
        a = z
        if activation.lower() == "relu":
            a[a < 0] = 0
            dg = a.copy()
            dg[dg > 0] = 1
            return {"a":a, "dg":dg}
        if activation.lower() == "sigmoid":
            a = 1/(1+np.exp(-a))
            dg = a*(1-a)
            return {"a":a, "dg":dg}
        if activation.lower() == "tanh":
            m = np.exp(a)
            n = np.exp(-1*a)
            a = (m-n)/(m+n)
            dg = 1 - a*a
            return {"a": a, "dg": dg}
        if activation.lower() == "softmax":
            a = np.exp(a)
            b = np.sum(a, axis=1, keepdims=True)
            a = a/b
            dg = a * (1 - a)
            return {"a": a, "dg": dg}


    def __loss_f(self,y_hat,y_true):
        """
        损失函数
        :param y_hat:
        :param y_true:
        :return:
        """
        if self.loss_func == "crossEntropy":
            loss = -1*np.sum(y_true*np.log(y_hat) + (1-y_true)*np.log(1-y_hat))/y_hat.shape[0]
            dz_hat = -(y_true - y_hat)

        elif self.loss_func == "mse":
            loss = np.sum(np.power(y_hat - y_true, 2))/(2*y_hat.shape[0])
            dz_hat = (y_hat - y_true)*self.dg[-1]

        # if self.regularization == "none":
        #     pass
        # elif self.regularization == "L1":
        #     w_count = 0
        #     mean = 0
        #     for w in self.w:
        #         mean += np.sum(np.abs(w))
        #         w_count += w.shape[0] * w.shape[1]
        #     mean = mean / w_count
        #     loss = loss + mean*self.regularization_rate
        # elif self.regularization == "L2":
        #     w_count = 0
        #     mean2 = 0
        #     for w in self.w:
        #         mean2 += np.sum(np.power(w, 2))
        #         w_count += w.shape[0] * w.shape[1]
        #     mean2 = mean2 / w_count
        #     loss = loss + mean2*self.regularization_rate

        return {"loss": loss, "dz_hat": dz_hat}


    def __regularization_f(self):
        """
        正则化
        :return:
        """
        if self.regularization == "none":
            pass
        elif self.regularization == "L1":
            for i in range(len(self.dw)):
                self.dw[i] += np.abs(self.regularization_rate)
        elif self.regularization == "L2":
            for i in range(len(self.dw)):
                self.dw[i] += np.abs(self.regularization_rate * self.dw[i])


    def __optimizer_f(self):
        pass

    def forward(self, test_x, test_y):
        """
        用训练好的模型前向传播
        :param test_x:
        :param test_y:
        :return:
        """
        a = []  # a = g(z)
        z = np.dot(test_x, self.w[0]) + self.b[0]
        aa = self.__activation_f(activation=self.activation_func, z=z)
        a.append(aa["a"])
        for l in range(1, self.layers_num):
            if l < self.layers_num - 1:
                activation = self.activation_func
            # 输出层
            elif l == self.layers_num - 1:
                activation = self.o_activation_func
            z = np.dot(a[l - 1], self.w[l]) + self.b[l]
            aa = self.__activation_f(activation=activation, z=z)
            a.append(aa["a"])
        rs = []
        for i in range(a[-1].shape[0]):
            rs.append(np.argmax(a[-1][i]))
        return np.asarray(rs)

    def fit(self, train_x, train_y, test_x, test_y, epoch=1):
        """
        训练函数
        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :param epoch:
        :return:
        """
        m = train_x.shape[0]  #样本数
        nd = np.random.RandomState(14)
        self.w.insert(0,nd.random([train_x.shape[1], self.layerStructure[0]])*0.01)  #添加第一层网络的训练参数w
        for epoch in range(epoch):      #训练轮数
            self.z = []                 #z = x * w.T
            self.a = []                 #a = g(z)
            self.dg = []                #dg = da/dz
            self.dz = []                #dz = dL/dz = dL/da * da/dz
            self.dw = []                #dw = dL/dw = dL/dz * dz/dw
            self.db = []                #db = dL/db = dL/dz * dz/db = dL/dz

            ###前向传播###
            z = np.dot(train_x, self.w[0]) + self.b[0]
            aa = self.__activation_f(activation=self.activation_func,z=z)
            self.z.append(z)
            self.a.append(aa["a"])
            self.dg.append(aa["dg"])
            for l in range(1,self.layers_num):
                if l < self.layers_num - 1:
                    activation = self.activation_func
                # 输出层
                elif l == self.layers_num - 1:
                    activation = self.o_activation_func
                z = np.dot(self.a[l-1], self.w[l]) + self.b[l]
                aa = self.__activation_f(activation=activation, z=z)
                self.z.append(z)
                self.a.append(aa["a"])
                self.dg.append(aa["dg"])

            ###后向传播###
            loss = self.__loss_f(y_hat=self.a[-1],y_true=train_y)
            print(loss["loss"])
            self.loss.append(loss["loss"])
            self.dz.insert(0, loss["dz_hat"])
            self.dw.insert(0, np.dot(self.a[-2].T, self.dz[0]) / m)
            self.db.insert(0, np.sum(self.dz[0], axis=0, keepdims=True) / m)
            for b in reversed(range(0, self.layers_num-1)):
                self.dz.insert(0, np.dot(self.dz[0], self.w[b+1].T)*self.dg[b])
                if b > 0:
                    self.dw.insert(0, np.dot(self.a[b-1].T, self.dz[0]) / m)
                elif b == 0:
                    self.dw.insert(0, np.dot(train_x.T, self.dz[0]) / m)
                self.db.insert(0, np.sum(self.dz[0], axis=0, keepdims=True) / m)

            ###参数更新###
            for i in range(self.dw.__len__()):
                self.__regularization_f() #正则化
                self.w[i] -= self.learn_rate * self.dw[i]
                self.b[i] = self.b[i] - self.learn_rate * self.db[i]



iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X / (np.max(X) - np.min(X)) * 0.99 + 0.01
oneHot = np.identity(3)
x_train, x_test, y_train, y_test = train_test_split(X, y)
y_train = oneHot[y_train]


m = model(layerStructure=(8,3))
m.fit(train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test, epoch=5000)

y_predict = m.forward(test_x=x_test, test_y=y_test)
print("真实值:",y_test)
print("预测值:",y_predict)
print("准确率:",sum(y_test == y_predict) / len(y_test))


#绘出loss曲线
plt.title("loss")
plt.xlabel("num")
plt.ylabel("loss")
plt.plot(m.loss, label= "$Loss$")
plt.legend()
plt.show()