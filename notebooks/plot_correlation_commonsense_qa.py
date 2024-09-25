# %%
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

methods = [ 
    r'$\mathrm{Ours}$',
    # r"$\mathrm{Self}$" + "-" + r"$\mathrm{influence}$",
    r'$\mathrm{Gradient~similarity}$',
    r'$\mathrm{Feature~similarity}$',
    r"$n$" + "-" + r"$\mathrm{gram~features}$",]
l1 = [0.6253643973732356]
l2 = [0.1591099893588]
l3 = [0.1000958391779287]
l4 = [0.0527008255473829]

N = 1
ind = np.arange(N) * 24  # the x locations for the groups
width = 2.0      # the width of the bars
shift = 3.0

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plt.subplots(figsize=(7, 6))
rects1 = ax.bar(ind + shift * 0, l4, width, color='slategrey', ecolor='white') # color='yellowgreen', ecolor='k', hatch="|"
rects2 = ax.bar(ind + width * 1 + shift, l3, width, color='mediumaquamarine', ecolor='white') # color='tomato', ecolor='k', hatch="x"
rects4 = ax.bar(ind + width * 2 + shift*2, l2, width, color='steelblue', ecolor='white')
rects5 = ax.bar(ind + width * 3 + shift*3, l1, width, color='coral', ecolor='white',) # hatch="\\"

#ax.set_ylim([0.2, 400])
ax.set_ylabel(r'$\mathrm{Spearsman~Corr.}$', fontsize=40)
# ax.set_xticks([ 0,  5.5, 11.2, 18.5])
# plt.xlim(-2, 17)
ax.xaxis.set_ticks_position('none') 
ax.set_xticklabels([], rotation = 20)
plt.yticks(np.arange(0.0, 0.9, 0.2))
# plt.ylim([0.48, 0.82])

# plt.legend([rects5, rects4, rects2, rects1], methods, fontsize=26, loc='upper left')

plt.tick_params(axis='x')
plt.title(r'$\mathrm{Commonsense~QA}$', fontsize=40)
ax.yaxis.grid(True, lw=0.8)

ax.tick_params(axis='both', which='major', labelsize=36)
ax.tick_params(axis='both', which='minor', labelsize=40)

plt.tight_layout()
plt.savefig('./figures/correlation_comparison_commonsense_qa.pdf', dpi=300, bbox_inches='tight', format='pdf')