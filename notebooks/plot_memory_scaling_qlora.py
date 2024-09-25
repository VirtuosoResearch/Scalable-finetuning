# %%
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

x = np.array([2.6, 8.0, 20.7, 27.2, 70.6])
y = np.array([2720, 6840, 13800, 17376, 41249])/1000
y_256 = np.array([4581, 9449, 21011, 26549, 65058])/1000
y_512 = np.array([6375, 12323, 28222, 35722, 88867])/1000
y_1024 = np.array([10161, 18703, 42644, 54068, 136486])/1000
# Create scatter plot
f, ax = plt.subplots(figsize=(12, 9))

scatter1 = ax.scatter(x, y, s=400,  marker='o')
l1 = ax.plot(x[:-1], y[:-1], linewidth=6, label=r'$\mathrm{Length=2}$')
ax.plot(x[-2:], y[-2:], linewidth=6, ls='--', color=l1[0].get_color())

scatter1 = ax.scatter(x, y_256, s=400,  marker='o')
l2 = ax.plot(x[:-2], y_256[:-2], linewidth=6, label=r'$\mathrm{Length=256}$')
ax.plot(x[-3:], y_256[-3:], linewidth=6, ls='--', color=l2[0].get_color())

scatter1 = ax.scatter(x, y_512, s=400,  marker='o')
l3 = ax.plot(x[:-3], y_512[:-3], linewidth=6, label=r'$\mathrm{Length=512}$')
ax.plot(x[-4:], y_512[-4:], linewidth=6, ls='--', color=l3[0].get_color())

scatter1 = ax.scatter(x[:2], y_1024[:2], s=400,  marker='o')
l4 = ax.plot(x[:2], y_1024[:2], linewidth=6, label=r'$\mathrm{Length=1024}$')

ax.hlines(24.576, 0, 100, color='red', linestyle='--', linewidth=6,)
# ax.hlines(48, 0, 100, color='red', linestyle='--', linewidth=6,)


ax.set_xlim(0, 100)
ax.set_ylim(-2, 50)
labels = [r'$\mathrm{Gemma}$' + "-" + r'$\mathrm{2B}$', 
          r'$\mathrm{Llama}$' + "-" + r'$\mathrm{3}$' + "-" + r'$\mathrm{8B}$', 
          r'$\mathrm{GPT}$' + "-" + r'$\mathrm{NeoX}$' + "-" + r'$\mathrm{20B}$', 
          r'$\mathrm{Gemma}$' + "-" + r'$\mathrm{27B}$' , # + "-" + r'$\mathrm{1.3B}$'
          r'$\mathrm{Llama}$' + "-" + r'$\mathrm{3}$' + "-" + r'$\mathrm{70B}$'] # 
plt.legend(fontsize=30)

# Label each point
for i, label in enumerate(labels):
    ax.text(x[i]+2, y[i], label, fontsize=38, ha='left', va='top', )
    
ax.text(60, 23, r'$\mathrm{RTX~6000~Memory}$', fontsize=38, ha='left', va='top', color='red')

# ax.text(60, 47, r'$\mathrm{RTX~A6000}$', fontsize=38, ha='left', va='top', color='red')


# plt.title('Scatter Plot with Labels')
ax.set_xlabel(r'$\mathrm{\#~Model~Parameters~(Billion)}$', fontsize=44)
ax.set_ylabel(r'$\mathrm{Memory~(GB)}$', fontsize=44)
ax.tick_params(labelsize=44)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/plot_scaling_of_memory.pdf", format="pdf", dpi=1200)
plt.show()