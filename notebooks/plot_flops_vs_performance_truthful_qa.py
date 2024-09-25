# %%
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


# Sample data points
x = [ 
5.80E+08,
8.98E+08,
1.04E+09,
1.04E+09,
5.54E+11,
5.04E+12,
2.61E+09,
1.53E+10
    ] 
y = [
0.234,
0.2400,
0.2448,
0.2495,
0.2772,
0.2821,
0.2725,
0.2793,
] 
x = np.array(x) # convert to numpy array
y = 100 - np.array(y)*100 # compute error rate

labels = [r'$\mathrm{MTL}$', 
          r'$\mathrm{DSIR}$', 
          r'$\mathrm{DEFT}$', 
          r'$\mathrm{LESS}$', 
          r'$\mathrm{FS}$',
          r'$\mathrm{RE}$',
          r'$\mathrm{\bf Grad}$' + r'$\mathrm{\bf Ex}$' + "-" r'$\mathrm{\bf FS}$',
          r'$\mathrm{\bf Grad}$' + r'$\mathrm{\bf Ex}$' + "-" r'$\mathrm{\bf RE}$',]

# Create scatter plot
f, ax = plt.subplots(figsize=(7.5, 7.5))
ax.scatter(x[:-2], y[:-2], color='wheat', s=500)

ax.scatter(x[-2], y[-2], color='forestgreen', s=3000, marker='*')
ax.scatter(x[-1], y[-1], color='red', s=3000, marker='*')
# ax.plot(x[-3:], y[-3:], color='orange', lw=4, linestyle='solid')
# Label each point
for i, label in enumerate(labels):
    if i in [0, 1]:
        ax.text(x[i]+0.8e9, y[i], label, fontsize=30, ha='left', va='bottom', )
    elif i in [2]:
        ax.text(x[i]+0.8e9, y[i]-0.3, label, fontsize=30, ha='left', va='bottom', )
    elif i in [3]:
        ax.text(x[i]+0.8e9, y[i]-0.2, label, fontsize=30, ha='left', va='top', )
    # elif i in [5]:
    #     ax.text(x[i], y[i]+0.3, label, fontsize=32, ha='left', va='bottom')
    elif i in [5]:
        ax.text(x[i]+5e12, y[i], label, fontsize=30, ha='left', va='bottom')
    elif i in [4]:
        ax.text(x[i]+5e11, y[i], label, fontsize=30, ha='left', va='bottom')
    elif i in [6]:
        ax.text(x[i], y[i]+0.5, label, fontsize=30, ha='left', va='bottom')
    else:
        ax.text(x[i], y[i]-0.5, label, fontsize=30, ha='left', va='top')

# set x-axis to log scale
plt.xscale('log')
plt.ylim(71, 78.3)
plt.yticks([72, 74, 76, 78])

plt.xticks([1E+09, 1E+10, 1E+11, 1E+12, 1E+13], [r'$10^9$', '', r'$10^{11}$', '', r'$10^{13}$'])
plt.xlim(0.2E+09, 8E+13)

plt.title(r'$\mathrm{TruthfulQA}$', fontsize=40)
plt.xlabel(r'$\mathrm{FLOPs}$', fontsize=40)
plt.ylabel(r'$\mathrm{Error~\%}$', fontsize=40)
ax.tick_params(labelsize=40)
# plt.title(r'$\mathrm{Dataset:~YouTube}$', fontsize=36)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/plot_flops_vs_performance_truthful.pdf", format="pdf", dpi=1200)
plt.show()