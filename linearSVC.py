# <span style="font-family:Microsoft YaHei;">'''
# LinearSVC 参数解释
# C：目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；
# loss ：指定损失函数
# penalty ：
# dual ：选择算法来解决对偶或原始优化问题。当n_samples > n_features 时dual=false。
# tol ：（default = 1e - 3）: svm结束标准的精度;
# multi_class：如果y输出类别包含多类，用来确定多类策略， ovr表示一对多，“crammer_singer”优化所有类别的一个共同的目标
# 如果选择“crammer_singer”，损失、惩罚和优化将会被被忽略。
# fit_intercept ：
# intercept_scaling ：
# class_weight ：对于每一个类别i设置惩罚系数C = class_weight[i]*C,如果不给出，权重自动调整为 n_samples / (n_classes * np.bincount(y))
# verbose：跟多线程有关，不大明白啥意思具体<pre name="code" class="python">

from sklearn.svm import SVC

X=[[0],[1],[2],[3]]
Y = [0,1,2,3]

clf = SVC(decision_function_shape='ovo') #ovo为一对一
clf.fit(X,Y)
print(clf.fit(X,Y))
dec = clf.decision_function([[1.8]])    #返回的是样本距离超平面的距离，为正最多的
print (dec)

clf.decision_function_shape = "ovr"
dec =clf.decision_function([1.8]) #返回的是样本距离超平面的距离，取最大的
print (dec)

#预测
print (clf.predict([1.8]))
