import numpy as np
import scipy
import tensorflow as tf
# import matplotlib.pyplot as plt

def auc(y_true, y_val, plot=True, fig_name='ct_train_dir/', accuracy=-1):
  #check input

  if len(y_true) != len(y_val):
      raise ValueError('Label vector (y_true) and corresponding value vector (y_val) must have the same length.\n')

  #empty arrays, true positive and false positive numbers
  tp = []
  fp = []
  y_true = y_true * 2 - 1 # between -1 and 1
  y_val = y_val *2 - 1

  #count 1's and -1's in y_true
  cond_positive = list(y_true).count(1)
  cond_negative = list(y_true).count(-1)
  #all possibly relevant bias parameters stored in a list
  bias_set = sorted(list(set(y_val)), key=float, reverse=True)
  bias_set.append(min(bias_set)*0.999)
  #originally *0.9 multiplier

  #initialize y_pred array full of negative predictions (-1)
  y_pred = np.ones(len(y_true))*(-1)

  #the computation time is mainly influenced by this for loop
  #for a contamination rate of 1% it already takes ~8s to terminate
  for bias in bias_set:
      # print(bias)
      # lower values tend to correspond to label -1
      #indices of values which exceed the bias
      posIdx = np.where(y_val > bias)
      #set predicted values to 1
      y_pred[posIdx] = 1
      #the following function simply calculates results which enable a distinction
      #between the cases of true positive and  false positive
      results = np.asarray(y_true) + 2 * np.asarray(y_pred)
      #append the amount of tp's and fp's
      tp.append(float(list(results).count(3)))
      fp.append(float(list(results).count(1)))

  #calculate false positive/negative rate
  tpr = np.asarray(tp)/cond_positive
  fpr = np.asarray(fp)/cond_negative
  #optional scatterplot

  #calculate AUC
  area_under = np.trapz(tpr,fpr)
  #title = 'AUC: ' + '%.3f' % (round(area_under,3)) + ' acc_at_50: ' + '%.3f' % round(accuracy,3)
  title = 'AUC: ' + '%.3f' % (round(area_under, 3))
  fig_name = fig_name + 'auc' + '%.3f' % (round(area_under, 3)) + '.png'
  print(title)
  if (area_under >= 0.56):
      plt.scatter(fpr,tpr)
      plt.title(title)  # subplot 211 title
      plt.ylabel('True Positive Rate')
      plt.xlabel('False Positive Rate')
      plt.hold(True)
      plt.plot([0, 0.5, 1.0], [0, 0.5, 1.0], '--')
      plt.axis([0, 1.0, 0, 1.0])
      plt.savefig(fig_name)
      plt.show()
      plt.close()

  return area_under, fpr, tpr