import numpy as np
import random as rn
import time


def SOA(val, objfun, x_min, x_max, itermax):

    global best_sub
    N,D = val.shape[0], val.shape[1]

    Mv = 2
    m = x_min[0]
    n = x_max[0]
    nv = np.random.rand(0, 2*3.14)

    x = val

    f = objfun(x)

    fgbest = min(f)
    igbest = np.where(min(f) == fgbest)
    gbest = x[igbest, :]
    pbest = x
    fpbest = f

    fbst = np.zeros((itermax, 1))
    ct = time.time()

    # Iterate
    for it in range(itermax):
        print(it)

        Ms = Mv - (it * Mv / itermax)
        S = Ms * x

        Cr = 0.5 * np.random.rand()
        L = Cr * (fgbest[it,:] - x[it,:])

        G = S + L

        a = np.radians(nv) * np.sin(nv)
        b = np.radians(nv) * np.cos(nv)
        c = np.radians(nv) * nv

        x[it,:] = (G *(a + b + c))* x[it]
        for mi in range(N):
            for mj in range(D):
                if x[mi, mj] < x_min[mi, mj]:
                    x[mi, mj] = x_min[mi, mj]
                else:
                    if x[mi, mj] > x_max[mi, mj]:
                        x[mi, mj] = x_max[mi, mj]
        high_ind = np.where(x < m)
        if len(high_ind[0]) != 0:
            x[high_ind] = m
        low_ind = np.where(x > n)
        if len(low_ind[0]) != 0:
            x[low_ind] = n
        f = objfun(x)
        minf = min(f)
        iminf = np.where(min(f) == minf)
        if minf <= fgbest:
            fgbest = minf
            gbest = x[iminf, :]
            best_sub = x[iminf, :]
            fbst[it] = minf
        else:
            fbst[it] = fgbest
            best_sub = gbest
        inewpb = np.where(f <= fpbest)
        pbest[inewpb, :] = x[inewpb[0], :]
        fpbest[inewpb] = f[inewpb]
    ct = time.time() - ct
    best_fit = fbst[itermax - 1]
    best_sub = np.reshape(best_sub, (-1))
    return best_fit, fbst, best_sub, ct

