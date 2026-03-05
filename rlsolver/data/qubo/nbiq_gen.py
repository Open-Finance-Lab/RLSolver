import numpy as np


def gen_data(nvar, neg_ratio, dens_ratio):
    data = np.random.randint(10, 101, (nvar, nvar))
    neg = np.random.uniform(size=(nvar, nvar))
    neg = (neg > neg_ratio)
    neg = neg * 2 - 1
    dens = np.random.uniform(size=(nvar, nvar))
    dens = (dens < dens_ratio)

    data = data * neg * dens
    data = np.triu(data)
    data = data + data.T
    for i in range(nvar):
        data[i][i] = data[i][i] / 2
    return data


nvar = 5000
neg_ratio = 0.499999
dens_ratio = 0.8
np.random.seed(1001)
for nvar in [5000, 7000, 10000]:
    cnt = 0
    for neg_ratio in [0.4995, 0.4999, 0.49995, 0.49999, 0.5]:
        print(nvar, neg_ratio)
        sum = -1
        while(sum < 0):
            data = gen_data(nvar, neg_ratio, dens_ratio)
            sum = data.sum()
            print(sum)
            print((data > 0).sum(), (data < 0).sum(), (data == 0).sum())
        cnt = cnt + 1
        np.save("nbiq_{}_{}.npy".format(nvar, cnt), data)
