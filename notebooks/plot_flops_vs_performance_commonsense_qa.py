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
1.19E+07,
1.78E+07,
1.84E+07,
1.84E+07,
5.98E+08,
5.98E+09,
2.03E+07,
3.71E+07,
    ] 
y = [
63.75,
64.56,
64.78,
64.92,
66.29,
66.70,
66.09,
66.50,
] 
x = np.array(x) # convert to numpy array
y = 100 - np.array(y) # compute error rate

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
        ax.text(x[i]+0.5e7, y[i]+0.15, label, fontsize=30, ha='left', va='bottom', )
    elif i in [2]:
        ax.text(x[i]+1e7, y[i]-0.1, label, fontsize=30, ha='left', va='bottom', )
    elif i in [3]:
        ax.text(x[i]+1e7, y[i]-0.15, label, fontsize=30, ha='left', va='top', )
    # elif i in [5]:
    #     ax.text(x[i], y[i]+0.3, label, fontsize=32, ha='left', va='bottom')
    elif i in [5]:
        ax.text(x[i]+2e9, y[i], label, fontsize=30, ha='left', va='bottom')
    elif i in [4]:
        ax.text(x[i]+4e8, y[i]-0.2, label, fontsize=30, ha='left', va='bottom')
    elif i in [6]:
        ax.text(x[i]+1e7, y[i]+0.2, label, fontsize=30, ha='left', va='bottom')
    else:
        ax.text(x[i]+1e7, y[i]-0.3, label, fontsize=30, ha='left', va='top')

# set x-axis to log scale
plt.xscale('log')
plt.ylim(32.8, 36.8)
plt.yticks([33, 34, 35, 36])

plt.xticks([1E+07, 1E+08, 1E+09, 1E+10])
plt.xlim(0.5E+07, 4E+10)

plt.title(r'$\mathrm{CommonsenseQA}$', fontsize=40)
plt.xlabel(r'$\mathrm{FLOPs}$', fontsize=40)
plt.ylabel(r'$\mathrm{Error~\%}$', fontsize=40)
ax.tick_params(labelsize=40)
# plt.title(r'$\mathrm{Dataset:~YouTube}$', fontsize=36)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/plot_flops_vs_performance_commonsense_qa.pdf", format="pdf", dpi=1200)
plt.show()