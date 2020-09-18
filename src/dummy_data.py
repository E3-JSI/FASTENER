'''
A dataset generator used for testing.
A dataset with 10^5 (N_samp) samples and 14 (N = 4 --> 4 + 4 + 3 + 2 + 1 = 14) features is generated.
Targets are binary (Y) and generated based on an arbitrary function.
'''

import numpy as np

np.random.seed(202000)
N_samp = 10 ** 5
N_test = int(N_samp * 0.8)
N = 4

XX = np.array([
    np.random.normal(0, j+1, N_samp) for j in range(N)
])

Y = XX[1] - 2 * XX[1] * XX[2] + 0.5 * XX[0] * XX[0] + 0.5 * np.random.normal(0, 1, N_samp)
labels = Y > 0
x = N

def a(i, j, s):
    global x
    x += 1
    return s

XXX = np.concatenate(
    [XX, np.concatenate(
        [(a(i, j, [XX[i] * XX[j]])) for i in range(N) for j in range(i, N)])]
).T

XX_train = XXX[:N_test, :]
XX_test = XXX[N_test:, :]
labels_train = labels[:N_test]
labels_test = labels[N_test:]