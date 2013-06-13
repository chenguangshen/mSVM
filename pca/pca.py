import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_svmlight_file
from pprint import pprint
import sklearn

# f = open("/Users/cgshen/workspace/mSVM/data/testcase_200.txt")

# n = int(f.readline())

# data = []
# label = []

# for i in range(n):
# 	num = int(f.readline())
# 	label.append(num)
# 	t = []
# 	for j in range(10):
# 		t.append(float(f.readline()))
# 	data.append(t)

# f.close()
# data = np.array(data)
# label = np.array(label)
# print data.shape

X_train, y_train = load_svmlight_file("/Users/cgshen/workspace/mSVM/data/vowel.scale.t")

data = np.array(X_train.todense())
label = np.array(y_train)
print data.shape
print label.shape
label = [int(t) for t in label]

pca = PCA(n_components=9)
data_reduced = pca.fit_transform(data)
print data_reduced.shape

sklearn.datasets.dump_svmlight_file(X=data_reduced, y=label, f="/Users/cgshen/workspace/mSVM/data/vowel.scale.t.pca9")
# libsvm = sklearn.svm.SVC(kernel='rbf', gamma=0.8, C=5, tol=0.5)