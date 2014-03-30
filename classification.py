# helper methods for classification

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


class NormalizerClassifier:
  def __init__(self, normalizer, clf):
    self.normalizer = normalizer
    self.clf = clf
    self.norm_state = None
  def fit(self, X):
    self.normalizer.fit(X)
    Z = self.normalizer.transform(X)
    self.clf.fit(Z)
  def predict(self, X2):
    Z2 = self.normalizer.transform(X2)
    return self.clf.predict(Z2)

def original_order_train_test_fraction_split(train_fraction, data, labels):
  Xtrain = data[:train_fraction,:]
  ytrain = labels[:train_fraction]
  Xtest = data[train_fraction:,:]
  ytest = labels[train_fraction:]
  return (Xtrain, ytrain, Xtest, ytest)
