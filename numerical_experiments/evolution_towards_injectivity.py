import mcbe
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

plot_lists6 = []
plot_lists10 = []
plot_lists27 = []
plot_lists6_removecr = []
plot_lists10_removecr = []
plot_lists27_removecr = []
for iter in tqdm(range(20)):
    W6 = mcbe.random_sphere(6,3)[0]
    W10 = mcbe.random_sphere(10,3)[0]
    W27 = mcbe.random_sphere(27,3)[0]
    alpha6, plot_data6 = mcbe.mcbe(W6,10000,plot=True,iter_plot=1000, return_plot_data=True)
    alpha6_removecr, plot_data6_removecr = mcbe.mcbe(W6,10000,plot=True,iter_plot=1000, return_plot_data=True, remove_covering_radius=True)
    alpha10, plot_data10 = mcbe.mcbe(W10,10000,plot=True,iter_plot=1000, return_plot_data=True)
    alpha10_removecr, plot_data10_removecr = mcbe.mcbe(W10,10000,plot=True,iter_plot=1000, return_plot_data=True, remove_covering_radius=True)
    alpha27, plot_data27 = mcbe.mcbe(W27,10000,plot=True,iter_plot=1000, return_plot_data=True)
    alpha27_removecr, plot_data27_removecr = mcbe.mcbe(W27,10000,plot=True,iter_plot=1000, return_plot_data=True, remove_covering_radius=True)
    plot_lists6.append(plot_data6)
    plot_lists10.append(plot_data10)
    plot_lists27.append(plot_data27)
    plot_lists6_removecr.append(plot_data6_removecr)
    plot_lists10_removecr.append(plot_data10_removecr)
    plot_lists27_removecr.append(plot_data27_removecr)

plot_lists6 = np.array(plot_lists6)
plot_lists10 = np.array(plot_lists10)
plot_lists27 = np.array(plot_lists27)
plot_lists6_removecr = np.array(plot_lists6_removecr)
plot_lists10_removecr = np.array(plot_lists10_removecr)
plot_lists27_removecr = np.array(plot_lists27_removecr)
means6 = plot_lists6.mean(axis=0)
means10 = plot_lists10.mean(axis=0)
means27 = plot_lists27.mean(axis=0)
means6_removecr = plot_lists6_removecr.mean(axis=0)
means10_removecr = plot_lists10_removecr.mean(axis=0)
means27_removecr = plot_lists27_removecr.mean(axis=0)
stds6 = plot_lists6.std(axis=0)
stds10 = plot_lists10.std(axis=0)
stds27 = plot_lists27.std(axis=0)
stds6_removecr = plot_lists6_removecr.std(axis=0)
stds10_removecr = plot_lists10_removecr.std(axis=0)
stds27_removecr = plot_lists27_removecr.std(axis=0)

plot_lists60 = []
plot_lists100 = []
plot_lists270 = []
plot_lists60_removecr = []
plot_lists100_removecr = []
plot_lists270_removecr = []


for iter in tqdm(range(20)):
    W60 = mcbe.random_sphere(60,30)[0]
    W100 = mcbe.random_sphere(100,30)[0]
    W270 = mcbe.random_sphere(270,30)[0]
    alpha60, plot_data60 = mcbe.mcbe(W60,10000,plot=True,iter_plot=1000, return_plot_data=True)
    alpha60_removecr, plot_data60_removecr = mcbe.mcbe(W60,10000,plot=True,iter_plot=1000, return_plot_data=True, remove_covering_radius=True)
    alpha100, plot_data100 = mcbe.mcbe(W100,10000,plot=True,iter_plot=1000, return_plot_data=True)
    alpha100_removecr, plot_data100_removecr = mcbe.mcbe(W100,10000,plot=True,iter_plot=1000, return_plot_data=True, remove_covering_radius=True)
    alpha270, plot_data270 = mcbe.mcbe(W270,10000,plot=True,iter_plot=1000, return_plot_data=True)
    alpha270_removecr, plot_data270_removecr = mcbe.mcbe(W270,10000,plot=True,iter_plot=1000, return_plot_data=True, remove_covering_radius=True)
    plot_lists60.append(plot_data60)
    plot_lists100.append(plot_data100)
    plot_lists270.append(plot_data270)
    plot_lists60_removecr.append(plot_data60_removecr)
    plot_lists100_removecr.append(plot_data100_removecr)
    plot_lists270_removecr.append(plot_data270_removecr)

plot_lists60 = np.array(plot_lists60)
plot_lists100 = np.array(plot_lists100)
plot_lists270 = np.array(plot_lists270)
plot_lists60_removecr = np.array(plot_lists60_removecr)
plot_lists100_removecr = np.array(plot_lists100_removecr)
plot_lists270_removecr = np.array(plot_lists270_removecr)
means60 = plot_lists60.mean(axis=0)
means100 = plot_lists100.mean(axis=0)
means270 = plot_lists270.mean(axis=0)
means60_removecr = plot_lists60_removecr.mean(axis=0)
means100_removecr = plot_lists100_removecr.mean(axis=0)
means270_removecr = plot_lists270_removecr.mean(axis=0)
stds60 = plot_lists60.std(axis=0)
stds100 = plot_lists100.std(axis=0)
stds270 = plot_lists270.std(axis=0)
stds60_removecr = plot_lists60_removecr.std(axis=0)
stds100_removecr = plot_lists100_removecr.std(axis=0)
stds270_removecr = plot_lists270_removecr.std(axis=0)


plt.figure(figsize=(20, 5))
plt.subplots_adjust(wspace=0.1)

plt.subplot(1, 2, 2, )

plt.plot(means60,label="n=30, m=60",marker="o",markevery=200, markersize=5)
plt.fill_between(np.arange(len(means60)),means60-stds60,[np.min([x,1]) for x in means60+stds60],alpha=0.2)
plt.plot(means100,label="n=30, m=100",marker="d",markevery=200,markersize=5)
plt.fill_between(np.arange(len(means100)),means100-stds100,[np.min([x,1]) for x in means100+stds100],alpha=0.2)
plt.plot(means270,label="n=30, m=270",marker="s",markevery=200,markersize=5)
plt.fill_between(np.arange(len(means270)),means270-stds270,[np.min([x,1]) for x in means270+stds270],alpha=0.2)
plt.xlabel("iteration") 
plt.grid(linestyle='--', alpha=0.5)
plt.legend(loc="lower right")
plt.ylim([0.2,1])
plt.xscale('log')

plt.subplot(1, 2, 1, )

plt.plot(means6,label="n=3, m=6",marker="o",markevery=200, markersize=5)
plt.fill_between(np.arange(len(means6)),means6-stds6,[np.min([x,1]) for x in means6+stds6],alpha=0.2)
plt.plot(means10,label="n=3, m=10",marker="d",markevery=200,markersize=5)
plt.fill_between(np.arange(len(means10)),means10-stds10,[np.min([x,1]) for x in means10+stds10],alpha=0.2)
plt.plot(means27,label="n=3, m=27",marker="s",markevery=200,markersize=5)
plt.fill_between(np.arange(len(means27)),means27-stds27,[np.min([x,1]) for x in means27+stds27],alpha=0.2)
plt.xlabel("iteration")
plt.ylabel("Proportion of point-wise injectivity   ")  
plt.grid(linestyle='--', alpha=0.5)
plt.legend(loc="lower right")
plt.ylim([0.2,1])
plt.xscale('log')
plt.show()