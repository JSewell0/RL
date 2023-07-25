import numpy as np
import matplotlib.pyplot as plt

def bandit(action):
    reward = reward_values[action]
    return reward

def update(action,reward,x):
 #   times_chosen[action] += 1    
    estimated_action_values[action] += 0.1*(reward-estimated_action_values[action])#(1/times_chosen[action])*(reward - estimated_action_values[action])
    x += np.random.normal(np.zeros(k),np.full(k,reward_variance))



epsilon = 0.1
k = 10
numruns = 1000
steps = 10000
reward_variance = 0.01
chosen_reward = np.array(np.zeros(steps))
for j in range(numruns):

    estimated_action_values = np.zeros(k)
    reward_values = np.zeros(k)
#    times_chosen = np.zeros(k)
    true_action_values = np.full(k,np.random.normal(0,1))

    for n in range(steps):

        rng = np.random.default_rng()
        reward_values = rng.normal(true_action_values,np.ones(k))

        action = 0;
        choice = rng.choice([0,1],p=[1-epsilon,epsilon])

        if choice == 0:
            action = np.where(estimated_action_values==max(estimated_action_values))[0][0]
        elif choice == 1:
            action = rng.integers(k)

        reward = bandit(action)
        chosen_reward[n] += reward  
        update(action,reward,true_action_values)

avg_reward = np.divide(chosen_reward,numruns)
fig, ax = plt.subplots()

ax.set_xlabel("steps")
#ax.set_xticks([1,250,500,750,1000])
ax.set_ylabel("avg reward")

ax.plot(np.linspace(0,steps,steps),avg_reward,"-",color="royalblue",label=f"\u03B5={epsilon}")
ax.legend()
plt.savefig("ns_const_stepsize(1e-1).png")
plt.show()
# print("\ntrue action value, estimated action value")
# for i in range(10):
#     print(f"action choice {i+1}: {true_action_values[i]:.2f}, {estimated_action_values[i]:.2f}\n")

# print(f"avg reward: {np.average(chosen_reward):.2f}") 
