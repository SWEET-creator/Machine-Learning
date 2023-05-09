from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import matplotlib.pyplot as plt

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X / 255, y.astype('int64'), stratify = y)

svm_model = svm.SVC(kernel='linear')

svm_model.fit(X_train, y_train)

y_predict = svm_model.predict(X_test)

ac_score = metrics.accuracy_score(y_test, y_predict)

print(ac_score)

#混同行列
fig = plt.figure()
cm = metrics.confusion_matrix(y_test, y_predict, labels=svm_model.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
plt.title("SVM (linear)")
disp.plot()

svm_model = svm.SVC(kernel='poly',degree=3)

svm_model.fit(X_train, y_train)

y_predict = svm_model.predict(X_test)

ac_score = metrics.accuracy_score(y_test, y_predict)

print(ac_score)

#混同行列
fig = plt.figure()
cm = metrics.confusion_matrix(y_test, y_predict, labels=svm_model.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
plt.title("SVM (poly degree=3)")
disp.plot()

svm_model = svm.SVC(kernel='poly',degree=4)

svm_model.fit(X_train, y_train)

y_predict = svm_model.predict(X_test)

ac_score = metrics.accuracy_score(y_test, y_predict)

print(ac_score)

#混同行列
fig = plt.figure()
cm = metrics.confusion_matrix(y_test, y_predict, labels=svm_model.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
plt.title("SVM (poly degree=4)")
disp.plot()


svm_model = svm.SVC(kernel='rbf')

svm_model.fit(X_train, y_train)

y_predict = svm_model.predict(X_test)

ac_score = metrics.accuracy_score(y_test, y_predict)

print(ac_score)

#混同行列
fig = plt.figure()
cm = metrics.confusion_matrix(y_test, y_predict, labels=svm_model.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
plt.title("SVM (RBF)")
disp.plot()