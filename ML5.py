import numpy
from sklearn import metrics
import matplotlib.pyplot as plt
#confusion matrix
actual = numpy.random.binomial(1, 0.9, size = 1000)
predicted = numpy.random.binomial(1, 0.9, size = 1000)
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
Accuracy = metrics.accuracy_score(actual, predicted)
print(Accuracy)
Precision = metrics.precision_score(actual, predicted)
print(Precision)
Sensitivity_recall = metrics.recall_score(actual, predicted)
print(Sensitivity_recall)
Specificity = metrics.recall_score(actual, predicted, pos_label=0)
print(Specificity)
F1_score = metrics.f1_score(actual, predicted)
print(F1_score)