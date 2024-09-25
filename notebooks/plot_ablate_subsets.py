# %%
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


f, ax2 = plt.subplots(figsize=(6, 5.5))


accs = np.array([62.59, 63.15, 63.64, 63.80, 63.90])
accs_std = np.array([0.43, 0.43, 0.34, 0.43, 0.31,])*0.4

x = np.arange(len(accs))

p1 = ax2.plot(x+1, accs, lw = 5, color="royalblue", linestyle="solid", label=r"$\mathrm{Task~}1$")
ax2.fill_between(
    x+1, 
    accs+accs_std,
    accs-accs_std, 
    color="royalblue", alpha=0.3
)

ax2.set_ylabel(r'$\mathrm{Accuracy~}(\%)$', fontsize = 40)
ax2.set_xlabel(r'$m$', fontsize = 40) # '+r'$\mathrm{~number~of~sampled~sets}
# ax.set_yticks(np.arange(0, 1.1, .2))
plt.xticks(x+1, [r'$200$', r'$400$', r'$600$', r'$800$', r'$1000$'])
ax2.set_yticks(np.arange(61, 65, 1))
ax2.set_ylim((61.5, 64.5))
# ax.set_xlim((-2.5, 3.5))

ax2.tick_params(labelsize=40)
ax2.grid(ls=':', lw=0.8)
# plt.legend(fontsize=20, loc=1)

plt.tight_layout()
plt.savefig("./figures/plot_ablate_subsets.pdf", format="pdf", dpi=1200)
plt.show()