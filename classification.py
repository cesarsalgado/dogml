import numpy as np
import cv2

def train_and_predict(clf, Ztrain, ytrain, Ztest):
  clf.fit(Ztrain, ytrain)
  predicted = clf.predict(Ztest)
  return predicted

class GlobalNormalizer:
  def fit(self, X):
    self.mean = X.mean()
    self.std = X.std()
  def transform(self, X2):
    return (X2 - self.mean)/self.std

class WrapperClassifier:
  def __init__(self):
    pass
  def __str__(self):
    return "%s:\n%s"%(self.__class__.__name__, self.clf.__str__())

class NormalizerClassifier(WrapperClassifier):
  def __init__(self, normalizer, clf):
    self.normalizer = normalizer
    self.clf = clf
    self.norm_state = None
  def fit(self, X, y):
    self.normalizer.fit(X)
    Z = self.normalizer.transform(X)
    self.clf.fit(Z, y)
  def predict(self, X2):
    Z2 = self.normalizer.transform(X2)
    return self.clf.predict(Z2)

class ConsecutiveFramesClassifier(WrapperClassifier):
  def __init__(self, clf):
    self.clf = clf
  def transform(self, X):
    return
  def fit(self, X, y):
    Z = self.transform(X)
    self.clf.fit(Z, y[1:])
  def predict(self, X2):
    Z2 = self.transform(X2)
    return self.clf.predict(Z2)

class ConcatenateTwoFramesClassifier(ConsecutiveFramesClassifier):
  def __init__(self, clf):
    ConsecutiveFramesClassifier.__init__(self, clf)
  def transform(self, X):
    n,d = X.shape
    return np.hstack((X[0:n-1,:], X[1:n,:]))

class DiffFramesClassifier(ConsecutiveFramesClassifier):
  def __init__(self, clf):
    ConsecutiveFramesClassifier.__init__(self, clf)
  def transform(self, X):
    n,d = X.shape
    return cv2.absdiff(X[0:n-1,:], X[1:n,:])

def original_order_train_test_fraction_split(train_fraction, data, labels):
  maxidx_train = int(train_fraction*len(labels))
  Xtrain = data[:maxidx_train,:]
  ytrain = labels[:maxidx_train]
  Xtest = data[maxidx_train:,:]
  ytest = labels[maxidx_train:]
  return (Xtrain, ytrain, Xtest, ytest)
