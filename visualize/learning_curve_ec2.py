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
with open('/home/smile/sac/result/wmc/darl1n/9_agents_2_ddl_400_iteration_16_seed/good_agent.pkl','rb') as f:
    reward_darl1n1 = pickle.load(f)

with open('/home/smile/sac/result/wmc/darl1n/9_agents_2_ddl_400_iteration_16_seed/global_time.pkl','rb') as f:
    gt1 = pickle.load(f)

with open('/home/smile/sac/result/wmc/darl1n/9_agents_2_ddl_400_iteration_25_seed/good_agent.pkl','rb') as f:
    reward_darl1n2 = pickle.load(f)

with open('/home/smile/sac/result/wmc/darl1n/9_agents_2_ddl_400_iteration_25_seed/global_time.pkl','rb') as f:
    gt2 = pickle.load(f)

with open('/home/smile/sac/result/wmc/sac/9_agents_2_ddl_30000_iteration_16_seed/reward.pkl','rb') as f:
    reward_sac1 = pickle.load(f)

with open('/home/smile/sac/result/wmc/sac/9_agents_2_ddl_30000_iteration_16_seed/time.pkl','rb') as f:
    gtsac1 = pickle.load(f)

with open('/home/smile/sac/result/wmc/sac/9_agents_2_ddl_30000_iteration_25_seed/reward.pkl','rb') as f:
    reward_sac2 = pickle.load(f)

with open('/home/smile/sac/result/wmc/sac/9_agents_2_ddl_30000_iteration_25_seed/time.pkl','rb') as f:
    gtsac2 = pickle.load(f)


plt.figure(figsize=(5.8,4.5))
plt.plot(gt1, reward_darl1n1, linestyle=':', linewidth=line_width, label='d1')
plt.plot(gt2, reward_darl1n2, linestyle=':', linewidth=line_width, label='d2')
plt.plot(gtsac1, reward_sac1, linestyle=':', linewidth=line_width, label='SAC1')
plt.plot(gtsac2, reward_sac2, linestyle=':', linewidth=line_width, label='SAC2')
plt.ylabel('Reward', fontsize=font_size)
plt.xlabel('Training iteration', fontsize=font_size)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(fontsize=font_size-2)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.subplots_adjust(bottom=0.15, left=0.2, top=0.95, wspace=0, hspace=0)
plt.grid()
plt.show()





# DARL1N 2 agents uncoded
with open('/home/smile/sac/result/wmc/darl1n/9_agents_10_ddl_400_iteration_16_seed/good_agent.pkl','rb') as f:
    reward_darl1n1 = pickle.load(f)

with open('/home/smile/sac/result/wmc/darl1n/9_agents_10_ddl_400_iteration_16_seed/global_time.pkl','rb') as f:
    gt1 = pickle.load(f)

with open('/home/smile/sac/result/wmc/darl1n/9_agents_10_ddl_400_iteration_25_seed/good_agent.pkl','rb') as f:
    reward_darl1n2 = pickle.load(f)

with open('/home/smile/sac/result/wmc/darl1n/9_agents_10_ddl_400_iteration_25_seed/global_time.pkl','rb') as f:
    gt2 = pickle.load(f)


with open('/home/smile/sac/result/wmc/sac/9_agents_10_ddl_30000_iteration_16_seed/reward.pkl','rb') as f:
    reward_sac1 = pickle.load(f)

with open('/home/smile/sac/result/wmc/sac/9_agents_10_ddl_30000_iteration_16_seed/time.pkl','rb') as f:
    gtsac1 = pickle.load(f)

with open('/home/smile/sac/result/wmc/sac/9_agents_10_ddl_30000_iteration_25_seed/reward.pkl','rb') as f:
    reward_sac2 = pickle.load(f)

with open('/home/smile/sac/result/wmc/sac/9_agents_10_ddl_30000_iteration_25_seed/time.pkl','rb') as f:
    gtsac2 = pickle.load(f)


plt.figure(figsize=(5.8,4.5))
plt.plot(gt1, reward_darl1n1, linestyle=':', linewidth=line_width, label='d1')
plt.plot(gt2, reward_darl1n2, linestyle=':', linewidth=line_width, label='d2')
plt.plot(gtsac1, reward_sac1, linestyle=':', linewidth=line_width, label='SAC1')
plt.plot(gtsac2, reward_sac2, linestyle=':', linewidth=line_width, label='SAC2')
plt.ylabel('Reward', fontsize=font_size)
plt.xlabel('Training iteration', fontsize=font_size)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(fontsize=font_size-2)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.subplots_adjust(bottom=0.15, left=0.2, top=0.95, wspace=0, hspace=0)
plt.grid()
plt.show()
