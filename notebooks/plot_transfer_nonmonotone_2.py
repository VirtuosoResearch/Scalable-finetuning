# %%
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

f, ax2 = plt.subplots(figsize=(7, 7))


pos = np.array([1.6840, 1.3040, 1.2695, 1.3400, 1.3709, 1.3683])
pos_std = np.array([0.0, 0.03171071579, 0.02171071579, 0.0248748813, 0.02460978303, 0.02115085241])

x = np.arange(len(pos))

# for i in range(len(x)):
#     scatter1 = ax2.scatter(x[i], pos[i], s=100, marker="o", edgecolors = "none", facecolors='orange')
# for i in range(len(x)):
#     scatter2 = ax2.scatter(x[i], pos_2[i], s=100, marker="D", edgecolors = "none", facecolors='orange')


p1 = ax2.plot(x+2, pos, lw = 5, color="royalblue", linestyle="dashed", label=r"$\mathrm{Task~}1$")
ax2.fill_between(
    x+2, 
    pos+pos_std,
    pos-pos_std, 
    color="royalblue", alpha=0.3
)

ax2.set_ylabel(r'$\mathrm{Target~task~loss}$', fontsize = 40)
ax2.set_xlabel(r'$\mathrm{Number~of~tasks~in}~S$', fontsize = 40) # '+r'$\mathrm{~number~of~sampled~sets}
# ax.set_yticks(np.arange(0, 1.1, .2))
plt.xticks(x+2,)
ax2.set_yticks(np.arange(1.2, 1.61, 0.2))
ax2.set_ylim((1.15, 1.7))
# ax.set_xlim((-2.5, 3.5))

ax2.tick_params(labelsize=40)
ax2.grid(ls=':', lw=0.8)
# plt.legend(fontsize=20, loc=1)

plt.tight_layout()
plt.savefig("./figures/plot_transfer_nonmonotonic_2.pdf", format="pdf", dpi=1200)
plt.show()