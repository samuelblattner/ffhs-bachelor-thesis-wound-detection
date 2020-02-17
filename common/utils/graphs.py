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


x=[x / 100 for x in range(100)]
print(x)
cross_entropy = [-np.log(_) for _ in x]
cross_entropy2 = [-np.log(1 - _) for _ in x]
focal_loss = [-(1-_)**2*np.log(_) for _ in x]
focal_loss2 = [-(_)**2*np.log(1 - _) for _ in x]
plt.plot(x, cross_entropy)
plt.plot(x, cross_entropy2)
plt.plot(x, focal_loss, linestyle='dashed')
plt.plot(x, focal_loss2, linestyle='dashed')
plt.xlabel('Predicted class probability of class c (p_c)')
plt.ylabel('Loss')
plt.legend((
    'True positive CE',
    'False positive CE',
    'True positive FL',
    'False positive FL',
))

plt.show()

with open('fig.pdf', 'wb') as f:
    fig.savefig(f, format='pdf')