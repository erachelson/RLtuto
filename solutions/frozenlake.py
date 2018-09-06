### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

# Let's initialize the Q-function
Qql = np.zeros((16,4))
max_steps = 5000000
gamma = 0.9
alpha = 0.001

def epsilon_greedy(Q, s, epsilon):
    a = np.argmax(Q[s,:])
    if(np.random.rand()<=epsilon): # random action
        aa = np.random.randint(env.action_space.n-1)
        if aa==a:
            a=env.action_space.n-1
        else:
            a=aa
    return a

# Q-learning
count = np.zeros((env.observation_space.n,env.action_space.n)) # to track update frequencies
epsilon = 1
x = env.reset()
for t in range(max_steps):
    if((t+1)%1000000==0):
        epsilon = epsilon/2
    a = epsilon_greedy(Qql,x,epsilon)
    y,r,d,_ = env.step(a)
    Qql[x][a] = Qql[x][a] + alpha * (r+gamma*np.max(Qql[y][:])-Qql[x][a])
    count[x][a] += 1
    if d==True:
        x = env.reset()
    else:
        x=y

# Q-learning's final value function and policy

print("Max error:", np.max(np.abs(Qql-Qstar)))
print("Final epsilon:", epsilon)
pi_ql = greedyQpolicy(Qql)
print("Greedy Q-learning policy:")
print_policy(pi_ql)
print("Difference between pi_sarsa and pi_star (recall that there are several optimal policies):")
print(pi_ql-pi_star)
Qpi_ql, residuals = policy_Qeval_iter(pi_ql,1e-4,10000)
print("Max difference in value between pi_sarsa and pi_star:", np.max(np.abs(Qpi_ql-Qstar)))
print("Min difference in value between pi_sarsa and pi_star:", np.min(np.abs(Qpi_ql-Qstar)))

# Plot visitation frequencies map

count_map = np.zeros((env.unwrapped.nrow, env.unwrapped.ncol, env.action_space.n))
for a in range(env.action_space.n):
    for x in range(env.observation_space.n):
        row,col = to_row_col(x)
        count_map[row, col, a] = count[x,a]

fig, axs = plt.subplots(ncols=4)
for a in range(env.action_space.n):
    name = "a = " + actions[a]
    axs[a].set_title(name)
    axs[a].imshow(np.log(count_map[:,:,a]+1), interpolation='nearest')
    #print("a=", a, ":", sep='')
    #print(count_map[:,:,a])
plt.show()
env.render()
