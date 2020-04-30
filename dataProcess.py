# python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/4/30 17:58
# @Author  : XjKai
# @FileName: dataProcess.py
# @Software: PyCharm

import numpy as np

def dataP(x, y, flatten=True, oneHot_num=False, normal_regularization=True, shuffle=True):
    """
    :param x:
    :param y:
    :param flatten: 是否拉直
    :param oneHot_num: 是否将y转为独热码，输入转换的长度
    :param normal_regularization: 是否归一化
    :param shuffle: 是否打乱顺序
    :return:
    """
    x_ = np.asarray([xx.flatten() for xx in x] if flatten else x, dtype=np.float32)
    y_ = np.asarray(np.identity(oneHot_num)[y.reshape(y.shape[0],)] if oneHot_num else y.reshape(y.shape[0],1))
    if normal_regularization:
        x_ -=  np.mean(x_)
        x_ /=  np.var(x_)
    if shuffle:
        shuffle_index = np.random.permutation(np.arange(len(x_)))
        x_ = x_[shuffle_index]
        y_ = y_[shuffle_index]

    return x_, y_


if __name__ == '__main__':
    pass