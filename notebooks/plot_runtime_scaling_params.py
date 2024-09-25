# %%
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

x = np.array([137, 248, 1100, 1400, 8000])
x = np.log(x)
y = np.array([3.171512648, 5.943787075, 17.60643097, 23.42866093, 120.3871058])
y = np.log(y)

# Create scatter plot
f, ax = plt.subplots(figsize=(8, 6))

scatter1 = ax.scatter(x, y, color='forestgreen', s=400,  marker='D')
ax.plot(x, y, color='forestgreen', linewidth=6)

ax.set_ylim(0.4, 5.6)
# ax.set_yticks(np.arange(0, 170, 40))
plt.xticks([4.91998093, 5.71342875, 6.90306546, 8.98719682],
           [r"$\mathrm{0.1}$", r"$\mathrm{0.3}$", r"$\mathrm{1}$", r"$\mathrm{8}$"])
plt.yticks([1, 2, 3, 4, 5 ],
           [r"$\mathrm{3}$", r"$\mathrm{6}$", r"$\mathrm{20}$", r"$\mathrm{60}$", r"$\mathrm{120}$"])

labels = [r'$\mathrm{GPT}$' + "-" + r'$\mathrm{2}$', 
          r'$\mathrm{FLAN}$' + "-" + r'$\mathrm{T5}$' , # + "-" + r'$\mathrm{Base}$' 
          r'$\mathrm{TinyLlama}$', # + "-" + r'$\mathrm{1.1B}$' 
          r'$\mathrm{GPT}$' + "-" + r'$\mathrm{Neo}$' , # + "-" + r'$\mathrm{1.3B}$'
          r'$\mathrm{Llama}$' + "-" + r'$\mathrm{3}$'] #  + "-" + r'$\mathrm{8B}$'

# Label each point
for i, label in enumerate(labels):
    if i in [0]:
        ax.text(x[i]+0.2, y[i], label, fontsize=38, ha='left', va='top', )
    elif i in [1]:
        ax.text(x[i]+0.2, y[i], label, fontsize=38, ha='left', va='top', )
    elif i in [2]:
        ax.text(x[i]-0.3, y[i]-0.2, label, fontsize=38, ha='right', va='bottom', )
    elif i in [3]:
        ax.text(x[i]-0.1, y[i]+0.2, label, fontsize=38, ha='right', va='bottom', )
    elif i in [4]:
        ax.text(x[i]-0.1, y[i], label, fontsize=38, ha='right', va='bottom')
    
# plt.title('Scatter Plot with Labels')
ax.set_xlabel(r'$\mathrm{\#~Model~Params~(Billion)}$', fontsize=44)
ax.set_ylabel(r'$\mathrm{GPU~hours}$', fontsize=44)
ax.tick_params(labelsize=44)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/plot_scaling_of_gpu_ours.pdf", format="pdf", dpi=1200)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

x = np.array([137, 248, 1100, 1400, 8000])
x = np.log(x)
y = np.array([145.9771398, 277.1515727, 792.7416745, 982.6279553, 5097.313894])
y = np.log(y)

# Create scatter plot
f, ax = plt.subplots(figsize=(8, 6))

scatter1 = ax.scatter(x, y, color='royalblue', s=400,  marker='D')
ax.plot(x, y, color='royalblue', linewidth=6, linestyle='dashed')

# ax.set_yticks(np.arange(0, 170, 40))
plt.xticks([4.91998093, 5.71342875, 6.90306546, 8.98719682],
           [r"$\mathrm{0.1}$", r"$\mathrm{0.3}$", r"$\mathrm{1}$", r"$\mathrm{8}$"])
plt.yticks([ 5,  6,  7,  8, 9],
           [r"$\mathrm{150}$", r"$\mathrm{300}$", r"$\mathrm{1000}$",  r"$\mathrm{3000}$", r"$\mathrm{6000}$"])
plt.ylim(4.3, 9.5)

labels = [r'$\mathrm{GPT}$' + "-" + r'$\mathrm{2}$', 
          r'$\mathrm{FLAN}$' + "-" + r'$\mathrm{T5}$' , # + "-" + r'$\mathrm{Base}$' 
          r'$\mathrm{TinyLlama}$', # + "-" + r'$\mathrm{1.1B}$' 
          r'$\mathrm{GPT}$' + "-" + r'$\mathrm{Neo}$' , # + "-" + r'$\mathrm{1.3B}$'
          r'$\mathrm{Llama}$' + "-" + r'$\mathrm{3}$'] #  + "-" + r'$\mathrm{8B}$'

# Label each point
for i, label in enumerate(labels):
    if i in [0]:
        ax.text(x[i]+0.2, y[i], label, fontsize=38, ha='left', va='top', )
    elif i in [1]:
        ax.text(x[i]+0.2, y[i], label, fontsize=38, ha='left', va='top', )
    elif i in [2]:
        ax.text(x[i]-0.3, y[i]-0.2, label, fontsize=38, ha='right', va='bottom', )
    elif i in [3]:
        ax.text(x[i]-0.1, y[i]+0.2, label, fontsize=38, ha='right', va='bottom', )
    elif i in [4]:
        ax.text(x[i]-0.1, y[i], label, fontsize=38, ha='right', va='bottom')
    
# plt.title('Scatter Plot with Labels')
ax.set_xlabel(r'$\mathrm{\#~Model~Params~(Billion)}$', fontsize=44)
# ax.set_ylabel(r'$\mathrm{GPU~hours}$', fontsize=48)
ax.tick_params(labelsize=44)

# plt.title(r'$\mathrm{Dataset:~RTE}$', fontsize=36)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/plot_scaling_of_gpu_full.pdf", format="pdf", dpi=1200)
plt.show()