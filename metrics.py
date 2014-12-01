from sklearn import metrics
import numpy as np

def confusion_measures(conf):
  result = np.zeros((3,3))
  result[:2,:2] = conf
  tn = result[0,0]
  fp = result[0,1]
  fn = result[1,0]
  tp = result[1,1]
  result[0,2] = float(tn)/(tn+fp) if (tn+fp) != 0 else 1.0
  result[1,2] = float(tp)/(fn+tp) if (fn+tp) != 0 else 1.0
  result[2,0] = float(tn)/(tn+fn) if (tn+fn) != 0 else 1.0
  result[2,1] = float(tp)/(fp+tp) if (fp+tp) != 0 else 1.0
  result[2,2] = float(tp+tn)/(tp+tn+fp+fn)
  return result

def get_confusion_measures(ytrue, ypred, nclasses=None):
  f1 = metrics.f1_score(ytrue, ypred)
  if nclasses != None:
    c = confusion_measures(metrics.confusion_matrix(ytrue, ypred, labels=range(nclasses)))
  else:
    c = confusion_measures(metrics.confusion_matrix(ytrue, ypred))
  horizontal_separator = 37*'-'
  result = bytearray()
  result.extend("%-18s %-22s %-14s   %-10s%-10s\n" %
      ("confusion matrix", "predicted false", "predicted true", "rates", "rates names"))
  result.extend("%-18s %s\n" % ("", horizontal_separator))
  result.extend("%-18s|%-22d|%-14d|  %-10.5f%-10s\n" %
      ("really false", c[0,0], c[0,1], c[0,2], "specificity"))
  result.extend("%-18s|%-22d|%-14d|  %-10.5f%-10s\n" %
      ("really true", c[1,0], c[1,1], c[1,2], "sensitivity or recall"))
  result.extend("%-18s %s\n" % ("", horizontal_separator))
  result.extend("%-18s %-22.5f %-14.5f   %-10.5f%-10s\n" %
      ("rates", c[2,0], c[2,1], c[2,2], "accuracy"))
  result.extend("%-18s %-22s %-14s   %-10s%-10s\n" %
      ("rates names", "neg. predictive val.", "precision", "accuracy", "f1=%.5f" % f1))
  return str(result)
