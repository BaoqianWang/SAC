import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.ticker as ticker
#import seaborn as sns; sns.set() # Because sns.set() generally does not need to be changed, it can be set by the way when importing the module
import csv
import matplotlib
import seaborn as sns;
matplotlib.rc('font', size=16)

np.random.seed(16)
line_width = 3
font_size = 20

# DARL1N 2 agents uncoded
with open('/home/smile/sac/result/wmc/darl1n/9agents_2000/good_agent.pkl','rb') as f:
    reward_darl1n = pickle.load(f)


# DARL1N 2 agents coded
with open('/home/smile/sac/result/wmc/sac/9_agents/10_ddl/reward.pkl','rb') as f:
    reward_sac = pickle.load(f)


# DARL1N 2 agents coded
with open('/home/smile/sac/result/wmc/sac/9_agents/10_ddl/wrong_reward.pkl','rb') as f:
    reward_sac_wrong = pickle.load(f)


plt.figure(figsize=(5.8,4.5))
plt.plot([(i+1)*200 for i in range(len(reward_sac))], reward_sac, linestyle=':', linewidth=line_width, label='SAC')
plt.plot([(i+1)*200 for i in range(len(reward_sac_wrong))], reward_sac_wrong, linestyle=':', linewidth=line_width, label='SAC Wrong')
plt.ylabel('Reward', fontsize=font_size)
plt.xlabel('Training iteration', fontsize=font_size)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(fontsize=font_size-2)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.subplots_adjust(bottom=0.15, left=0.2, top=0.95, wspace=0, hspace=0)
plt.grid()
plt.savefig('../figures/sac0308', transparent = False)
plt.show()


plt.figure(figsize=(5.8,4.5))
plt.plot([(i+1)*10 for i in range(len(reward_darl1n))], reward_darl1n, linestyle=':', linewidth=line_width, label='DARL1N')
plt.ylabel('Reward', fontsize=font_size)
plt.xlabel('Training iteration', fontsize=font_size)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(fontsize=font_size-2)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.subplots_adjust(bottom=0.15, left=0.2, top=0.95, wspace=0, hspace=0)
plt.grid()
plt.savefig('../figures/darl1n0308', transparent = False)
plt.show()
