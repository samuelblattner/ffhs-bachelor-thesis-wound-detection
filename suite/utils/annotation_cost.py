from mpl_toolkits.mplot3d import Axes3D
import numpy as np
res = 1000
P = range(1, res)
tau = 20
alpha = 30

beta = 5

first_valid_r = None
first_valid_f1 = None
last_valid_r = None
last_valid_f1 = None


from matplotlib import pyplot as plt


def annotation_cost(alpha, beta, tau, p):

    global first_valid_r, first_valid_f1, last_valid_r, last_valid_f1

    if alpha+beta-beta/p == 0:
        return 1, 1

    r = (-tau+alpha)/(alpha+beta-beta/p)
    capped_r = min(1.0, max(0.0, r))

    f1 = 2 * (p*r)/(p + r)
    print(p, r, f1, 'R-' if 0 < p < beta/(alpha + beta) else 'R+')

    if first_valid_r is None and 0 <= r <= 1.0:
        first_valid_r = p
        first_valid_f1 = f1

    if 0 <= r <= 1.0:
        last_valid_r = p
        last_valid_f1 = f1

    return f1, r


F1 = []
R = []
for p in P:

    f1, r = annotation_cost(alpha, beta, tau, p/res)
    F1.append(f1)
    R.append(r)

fig = plt.figure()
p = [p/res for p in P]

print('Min F1: ', min(F1))

plt.plot(p, R)
plt.plot(p, F1)
# ax.plot_surface(P, R, F1)
plt.xlabel('Precision')
plt.hlines(1.0, 0, 1.0, linestyles=('dashed',))
plt.hlines(0.0, 0, 1.0, linestyles=('dashed',))
# plt.vlines(first_valid_r, -0.4, 1.1, linestyles=('dashed',))

print(last_valid_r)
plt.vlines(last_valid_r, -0.4, 1.1, linestyles=('dashed',))
plt.xlim((0.0, 1.1))
plt.ylim((-0.3, 1.1))

annot_x = last_valid_r
annot_y_r = 1.0
annot_y_f1 = last_valid_f1

plt.annotate('p: {}, r: {:.04f}'.format(annot_x, 1.0), (annot_x, 1.0), (annot_x + 0.2, 1.0 - 0.2), arrowprops={'arrowstyle': '-', })
plt.annotate('p: {}, f1: {:.04f}'.format(annot_x, annot_y_f1), (annot_x, annot_y_f1), (annot_x + 0.2, annot_y_f1- 0.2), arrowprops={'arrowstyle': '-', })

plt.legend(('recall', 'F1'))
plt.show()

with open('fig.pdf', 'wb') as f:
    fig.savefig(f, format='pdf')

