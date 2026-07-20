from FileRead import readcol, writecol
import numpy as np
import sys
import matplotlib.pyplot as plt

[x, y] = readcol("filter_up_orig.txt", 'ff')

plt.plot(x, y)
inds = np.where(x > 450.)
y[inds] *= float(sys.argv[1])

plt.plot(x, y)
plt.yscale('log')
plt.ylim(1e-6, 2)
plt.savefig("u_scale=%.3g.pdf" % float(sys.argv[1]))
plt.close()

writecol("filter_up.txt", [x, y])
