# python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/4/30 21:10
# @Author  : XjKai
# @FileName: optimizers.py
# @Software: PyCharm
import numpy as np

def optimizer_f(dw, db, optimizer, learn_rate, global_step=None, beta=None, beta1=None):
    #SGD、SGDM、Adam、RMSProp、Adagrad
    # 一阶动量mt
    # 二阶动量Vt
    # 下降梯度yita = lr * mt / sqrt(Vt)

    m_w, m_b = 0, 0
    V_w, V_b = 1, 1
    beta = beta if beta != None else 0.9
    beta1 = beta1 if beta1 != None else 0.999

    if optimizer == "SGDM":
        m_w = beta * m_w + (1 - beta) * dw
        m_b = beta * m_b + (1 - beta) * db
        V_w = 1
        V_b = 1
    elif optimizer == "RMSProp":
        m_w = dw
        m_b = db
        V_w = beta * V_w + (1 - beta) * np.square(dw)
        V_b = beta * V_b + (1 - beta) * np.square(db)
    elif optimizer == "Adagrad":
        m_w = dw
        m_b = db
        V_w += np.square(dw)
        V_b += np.square(db)
    elif optimizer == "Adam":
        m_w = beta * m_w + (1 - beta) * dw
        m_b = beta * m_b + (1 - beta) * db
        V_w = beta1 * V_w + (1 - beta1) * np.square(dw)
        V_b = beta1 * V_b + (1 - beta1) * np.square(db)
        #修正后的一阶和二阶动量
        m_w = m_w/(1 - np.power(beta, int(global_step)))
        m_b = m_b/(1 - np.power(beta, int(global_step)))
        V_w = V_w/(1 - np.power(beta1, int(global_step)))
        V_b = V_b/(1 - np.power(beta1, int(global_step)))
    else:   #SGD
        m_w = dw
        m_b = db
        V_w = 1
        V_b = 1

    yita_w = learn_rate * (m_w / np.sqrt(V_w))
    yita_b = learn_rate * (m_b / np.sqrt(V_b))
    return yita_w, yita_b

