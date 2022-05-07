
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


num = 10

def LoadData(): #载入数据集
    data = []
    label = []
    path_cwd = "../ORL_dataset/"
    for j in range(1, 41):
        path = path_cwd + 's' + str(j)
        for number in range(1,num+1):
            path_full = path + '/' + str(number) +'.bmp'
            # print(path_full)
            image = Image.open(path_full).convert('L')
            # print(type(img))
            # image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            # print(image)
            # print(type(image))
            img = np.array(image)
            # print("img",img.shape)
            data.extend(img)
        label.extend(np.ones(num, dtype=np.int) * j)
    data = np.reshape(data, (num*j, 112*92))
    # print(np.matrix(data).shape)
    # print(np.matrix(label).T.shape)
    return np.matrix(data), np.matrix(label).T       #返回数据和标签



def knn(neighbor, traindata, trainlabel, testdata):
    neigh = KNeighborsClassifier(n_neighbors=neighbor)
    neigh.fit(traindata, trainlabel)
    return neigh.predict(testdata)

if __name__ == '__main__':

    # 设置pca保留数据方差值和k
    var,k = 0.75, 1
    Data_train, Data_test, Label_train, Label_test = train_test_split(*LoadData())
    pca = PCA(var, True, True)  # 建立pca类，设置参数，保留90%的数据方差
    trainDataS = pca.fit_transform(Data_train)  # 拟合并降维训练数据
    # print(len(Data_test))

    acc = 0
    num_test = len(Data_test)

    for i in range(len(Data_test)):
        testDataS = pca.transform(Data_test[i].ravel())
        
        result = knn(k,trainDataS,Label_train,testDataS)
        # print("预测:",result[0])
        # print("实际:",int(Label_test[i]))
        if result[0] == int(Label_test[i]):
            acc += 1
    accuracy = float(acc/num_test*100)
    print("var={0},\tk={1}".format(var,k))
    print("accuracy:\t{0}%".format(accuracy))




    