import math

from matplotlib import pyplot as plt
import numpy as np

# prec_recall = np.array([
#     [0.0, 0.2, 0.4, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8, 0.8, ],
#     [0.0, 0.5, 0.66, 0.5, 0.4, 0.5, 0.57, 0.5, 0.44, 0.4, ],
# ])
#
# prec_recall2 = np.array([
#     [0.2, 0.4, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
#     [1.0, 1.0, 1.0, 0.75, 0.8, 0.67, 0.57, 0.5, 0.44, 0.4],
# ])
#
from pandas import DataFrame
from scipy import stats

fig = plt.figure(figsize=(6,4))
# plt.plot(prec_recall[0], prec_recall[1])
#
# plt.plot(prec_recall2[0], prec_recall2[1])
#
# plt.ylim(0, 1.2)
# plt.xlim(0, 1.0)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
#
# plt.grid()
# plt.show()
#
# with open('fig.pdf', 'wb') as f:
#     fig.savefig(f, format='pdf')


# x=[x / 100 for x in range(100)]
# print(x)
# cross_entropy = [-np.log(_) for _ in x]
# cross_entropy2 = [-np.log(1 - _) for _ in x]
# focal_loss = [-(1-_)**2*np.log(_) for _ in x]
# focal_loss2 = [-(_)**2*np.log(1 - _) for _ in x]
# plt.plot(x, cross_entropy)
# plt.plot(x, cross_entropy2)
# plt.plot(x, focal_loss, linestyle='dashed')
# plt.plot(x, focal_loss2, linestyle='dashed')
# plt.xlabel('Predicted class probability of class c (p_c)')
# plt.ylabel('Loss')
# plt.legend((
#     'True positive CE',
#     'False positive CE',
#     'True positive FL',
#     'False positive FL',
# ))
#
# plt.show()
#
# with open('fig.pdf', 'wb') as f:
#     fig.savefig(f, format='pdf')



# ===========================
# df = DataFrame.from_dict({
#     'Retina 50':         [1.06, 0.02, 0.21, 1.21, 0.69, 3.48, 0.00, 0.32, 0.96, 3.17, 2.55, 1.92, 5.42, 7.35, 5.03, 3.21, 2.49, 0.55, 0.24, 0.95, 0.95, 2.11, 2.11, 2.37, 2.90, 2.91, 6.84],
#     'Retina 152':        [0.04, 0.02, 0.96, 0.07, 0.00, 0.00, 0.00, 0.16, 0.00, 0.00, 0.96, 0.96, 7.82, 5.95, 9.99, 6.43, 5.98, 0.06, 0.00, 0.20, 0.20, 0.38, 1.47, 1.47, 2.24, 8.27, 5.34],
#     'Retina 50, Cases':  [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 2.25, 0.60, 2.82, 0.00, 1.93, 1.93, 1.83, 1.45, 1.47, 3.09],
#     'Retina 152, Cases': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.18, 0.00, 1.40, 1.40, 3.11, 0.48, 2.02, 1.36, 2.60, 5.70]
# }, orient='index',
# columns=[
#     '1001', '1002', '1003', '1010a', '1010b', '1010c', '1011a', '1011b', '1011c', '1012a', '1012b', '1012c', '1020',
#     '1021', '1030', '1031', '1033', '1100', '1101', '1110', '1111', '1112', '1113', '1200', '1201', '1300', '1301'
# ])
#
# plt.figure(figsize=(10, 2))
# df = df.transpose()
# df.plot.bar()
# plt.show()


# =========================
# mu = 1
# variance = 4
# variance2 = 3
# variance3 = 2
# sigma = math.sqrt(variance)
# sigma=4
# sigma2=3
# sigma3=2
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# x2 = np.linspace(mu - 3*sigma2, mu + 3*sigma2, 100)
# x3 = np.linspace(mu - 3*sigma3, mu + 3*sigma3, 100)
# plt.plot(x, stats.norm.pdf(x, mu, sigma))
# plt.plot(x2, stats.norm.pdf(x, mu, sigma))
# plt.plot(x3, stats.norm.pdf(x, mu, sigma))
# plt.show()


from imgaug.parameters import show_distributions_grid, Clip, Normal, Absolute, Add
show_distributions_grid([
    Add(Absolute(Normal(1.0, 2.0)), 1),
],graph_sizes=(2048, 2048), )