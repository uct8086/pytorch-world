import torch
from torch.distributions import multinomial
import sys
sys.path.append("..")
from d2l.torch import set_figsize, plt


fair_probs = torch.ones([6]) / 6
# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample((500,))
# counts / 1000 # 相对频率作为估计值

# print(counts, counts/1000)
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
set_figsize((6, 4.5))
for i in range(6):
    plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
plt.axhline(y=0.167, color='black', linestyle='dashed')
plt.gca().set_xlabel('Groups of experiments')
plt.gca().set_ylabel('Estimated probability')
plt.legend()
plt.show()