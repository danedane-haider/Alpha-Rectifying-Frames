import matplotlib.pyplot as plt
import numpy as np
import mcbe


#################### compute the maximal biases for random ReLU layers ####################

#m_max = 150
#n_max = 30
#b0_inj = np.zeros((m_max,m_max))
#bnorm_inj = np.zeros((m_max,m_max))
#bnorm_injr2 = np.zeros((m_max,m_max))

#long runtime!!
#m_range = range(2,m_max)

#for m in tqdm(m_range):
    #for n in range(2,np.min([m,n_max])):
        #W = mcbe.random_point(m,n)#
        #alpha = mcbe.mcbe(W,500000,distribution="normal")
        
        #b0_inj[n,m] = np.mean(np.array(alpha) >= 0)
        
        #bnorm = mcbe.get_point(distribution="normal",d=m)
        #bnorm_inj[n,m] = np.mean(np.array(alpha) >= bnorm)
        #bnorm_injr2[n,m] = np.mean(np.array(alpha_r2) >= bnorm)

# load precomputed biases:
alphas_normal = np.load("alphas_normal_bigger.npy")
alphas_normal = np.array(alphas_normal)


m_max = alphas_normal.shape[0]
n_max = alphas_normal.shape[1]

b0 = np.zeros((m_max, n_max))
bnormal = np.zeros((m_max, n_max))
bnormal_std1 = np.zeros((m_max, n_max))

rho = 0.005*(500000/np.log(500000))**(np.arange(2,n_max+2,dtype=float)**(-1))*np.arange(2,n_max+2,dtype=float)

for m in range(2,m_max):
    for n in range(n_max):

        alphas_normal[m, n, :] = alphas_normal[m, n, :] - rho[n]

        #zero bias
        b0[m, n] = np.mean(alphas_normal[m, n, :][:m] >= 0)
        if m < 2*n:
            b0[m, n] = 0

        #std 0.1
        normalbias = np.random.normal(loc=0, scale=0.1, size=m)
        bnormal[m, n] = np.mean(alphas_normal[m, n, :][:m] >= normalbias)
        if np.all(alphas_normal[m, n, :] == - rho[n]):
            bnormal[m, n] = 0

        #std 1
        normalbias = np.random.normal(loc=0, scale=1, size=m)
        bnormal_std1[m, n] = np.mean(alphas_normal[m, n, :][:m] >= normalbias)
        if np.all(alphas_normal[m, n, :] == - rho[n]):
            bnormal_std1[m, n] = 0



fig, axs = plt.subplots(1, 3, figsize=(25,8), sharey=True)
plt.rcParams.update({'font.family':'Times New Roman', 'font.size': 28})

axs[0].imshow(b0,cmap="cividis")
axs[0].set_xlabel("dimension (n)")
axs[0].set_ylim((2.5,115))
#axs[0].plot(range(m_max),[x*9 for x in range(int(m_max))],linewidth=3,linestyle="--",color = "deepskyblue")
axs[0].plot(range(m_max),[x*6.7 for x in range(int(m_max))],linewidth=3,color="magenta")
#axs[0].plot(range(m_max),[x*3.3 for x in range(int(m_max))],linewidth=3,linestyle="-.",color="red")
axs[0].plot(range(m_max),[x*2 for x in range(int(m_max))],linewidth=3,linestyle="--",color="limegreen")
axs[0].set_xlim((1.5,n_max-0.5))
axs[0].set_aspect('auto')
axs[0].set_ylabel("number of elements (m)")
axs[0].legend(["m=6.7n","m=2n"],loc="lower right")

axs[1].imshow(bnormal,cmap="cividis")
axs[1].set_xlabel("dimension (n)")
axs[1].set_ylim((2.5,115))
#axs[1].plot(range(m_max),[x*9 for x in range(int(m_max))],linewidth=3,linestyle="--",color = "deepskyblue")
axs[1].plot(range(m_max),[x*6.7 for x in range(int(m_max))],linewidth=3,color="magenta")
#axs[1].plot(range(m_max),[x*3.3 for x in range(int(m_max))],linewidth=3,linestyle="-.",color="red")
axs[1].plot(range(m_max),[x*2 for x in range(int(m_max))],linewidth=3,linestyle="--",color="limegreen")
axs[1].set_xlim((1.5,n_max-0.5))
axs[1].set_aspect('auto')

axs[2].imshow(bnormal_std1,cmap="cividis")
axs[2].set_xlabel("dimension (n)")
axs[2].set_ylim((2.5,115))
#axs[2].plot(range(m_max),[x*9 for x in range(int(m_max))],linewidth=3,color = "deepskyblue",linestyle="--")
axs[2].plot(range(m_max),[x*6.7 for x in range(int(m_max))],linewidth=3,color="magenta")
#axs[2].plot(range(m_max),[x*3.3 for x in range(int(m_max))],linewidth=3,color="red",linestyle="-.")
axs[2].plot(range(m_max),[x*2 for x in range(int(m_max))],linewidth=3,color="limegreen",linestyle="--")
axs[2].set_xlim((1.5,n_max-0.5))
axs[2].set_aspect('auto')

plt.tight_layout()
plt.show()