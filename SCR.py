from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0.5, 1.5]
clf = svm.SVR()
clf.fit(X, y)
result = clf.predict([2, 2])#输出回归的结果
print(result)
