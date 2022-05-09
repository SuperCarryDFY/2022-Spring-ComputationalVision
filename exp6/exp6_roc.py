# *_*coding:utf-8 *_*

import os
import sys
import cv2
import numpy as np
from sklearn.metrics import roc_curve
from random import shuffle
import  matplotlib.pyplot as plt 

# def load_data_set(logger):
def load_data_set():
    '''
    导入数据集
    :param logger: 日志信息打印模块
    :return pos: 正样本文件名的列表
    :return neg: 负样本文件名的列表
    :return test: 测试数据集文件名的列表。
    '''
    pwd = os.getcwd()
    # 提取正样本
    pos_dir = os.path.join(pwd, 'Positive')
    if os.path.exists(pos_dir):
        pos = os.listdir(pos_dir)

    # 提取负样本
    neg_dir = os.path.join(pwd, 'Negative')
    if os.path.exists(neg_dir):
        neg = os.listdir(neg_dir)

    # 提取测试集
    test_dir = os.path.join(pwd, 'TestData')
    if os.path.exists(test_dir):
        # logger.info('Test data path is:{}'.format(test_dir))
        test = os.listdir(test_dir)
        # logger.info('Test samples number:{}'.format(len(test)))

    return pos, neg, test

def load_train_samples(pos, neg):
    '''
    合并正样本pos和负样本pos,创建训练数据集和对应的标签集
    :param pos: 正样本文件名列表
    :param neg: 负样本文件名列表
    :return samples: 合并后的训练样本文件名列表
    :return labels: 对应训练样本的标签列表
    '''
    pwd = os.getcwd()
    pos_dir = os.path.join(pwd, 'Positive')
    neg_dir = os.path.join(pwd, 'Negative')

    samples = []
    labels = []
    for f in pos:
        file_path = os.path.join(pos_dir, f)
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(1.)

    for f in neg:
        file_path = os.path.join(neg_dir, f)
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(-1.)

    # labels 要转换成numpy数组，类型为np.int32
    labels = np.int32(labels)
    labels_len = len(pos) + len(neg)
    labels = np.resize(labels, (labels_len, 1))

    return samples, labels

def extract_hog(samples):
    '''
    从训练数据集中提取HOG特征,并返回
    :param samples: 训练数据集
    :param logger: 日志信息打印模块
    :return train: 从训练数据集中提取的HOG特征
    '''
    train = []
    # logger.info('Extracting HOG Descriptors...')
    num = 0.
    total = len(samples)
    for f in samples:
        num += 1.
        # logger.info('Processing {} {:2.1f}%'.format(f, num/total*100))
        hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
        # hog = cv2.HOGDescriptor()
        img = cv2.imread(f, -1)
        img = cv2.resize(img, (64,128))
        descriptors = hog.compute(img)
        # logger.info('hog feature descriptor size: {}'.format(descriptors.shape))    # (3780, 1)
        train.append(descriptors)

    train = np.float32(train)
    train = np.resize(train, (total, 3780))

    return train

def get_svm_detector(svm):
    '''
    导出可以用于cv2.HOGDescriptor()的SVM检测器,实质上是训练好的SVM的支持向量和rho参数组成的列表
    :param svm: 训练好的SVM分类器
    :return: SVM的支持向量和rho参数组成的列表,可用作cv2.HOGDescriptor()的SVM检测器
    '''
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)

# def train_svm(train, labels, logger):
def train_svm(train, labels):
    '''
    训练SVM分类器
    :param train: 训练数据集
    :param labels: 对应训练集的标签
    :param logger: 日志信息打印模块
    :return: SVM检测器(注意:opencv的hogdescriptor中的svm不能直接用opencv的svm模型,而是要导出对应格式的数组)
    '''
    # logger.info('Configuring SVM classifier.')
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier
    svm.setType(cv2.ml.SVM_EPS_SVR)

    # logger.info('Starting training svm.')
    svm.train(train, cv2.ml.ROW_SAMPLE, labels)
    # logger.info('Training done.')

    pwd = os.getcwd()
    model_path = os.path.join(pwd, 'svm.xml')
    svm.save(model_path)
    # logger.info('Trained SVM classifier is saved as: {}'.format(model_path))

    return get_svm_detector(svm)


def draw_roc(svm_detector):
    hog = cv2.HOGDescriptor()
    # hog.setSVMDetector(svm_detector)
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    pwd = os.getcwd()

    test_dir = []
    # 提取正样本
    pos_dir = os.path.join(pwd, 'Positive')
    pos_files = os.listdir(pos_dir)
    print(pos_files)
    shuffle(pos_files)
    for file in pos_files[:50]:
        test_dir.append(os.path.join(pos_dir, file))

    # 提取负样本
    neg_dir = os.path.join(pwd, 'Negative') 
    neg_files = os.listdir(neg_dir)
    shuffle(neg_files)
    for file in neg_files[:50]:
        test_dir.append(os.path.join(neg_dir, file))
    

    label1 = [1 for _ in range(50)]
    label2 = [0 for _ in range(50)]
    label_dir = label1+label2
    # test 包含测试图片和标签
    test = list(zip(test_dir,label_dir))
    shuffle(test)

    test_dir[:],label_dir[:]= zip(*test)
    res = []
    for i in range(100):
        img = cv2.imread(test_dir[i])
        rects, _ = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)
        print(rects)
        if rects != ():
            res.append(1)
        else:
            res.append(0)
    fpr, tpr, thresholds= roc_curve(res, label_dir, pos_label=1)
    
    plt.plot(fpr, tpr)  
    plt.title('ROC_curve')  
    plt.ylabel('True Positive Rate')  
    plt.xlabel('False Positive Rate')
    plt.show()  


if __name__ == '__main__':
    # logger = logger_init()
    pos, neg, test = load_data_set()

    samples, labels = load_train_samples(pos, neg)
    
    train = extract_hog(samples)
    svm_detector = train_svm(train, labels)
    draw_roc(svm_detector)
