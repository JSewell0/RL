import numpy as np
import matplotlib.pyplot as plt
import pickle

class control:
    def __init__(self, alpha=0.05, n=5, delta=0.025, data={}, epsilon=0.1, q={}, pi={}, state_tracker={}, reward_tracker=np.array([]),mu=40,sigma=20,scale=1000):
        self.alpha = alpha #update step size param
        self.n = n #number of time steps before updating
        self.delta = delta #slope upper limit to check for terminal state
        self.data = data #stores state,action, and reward per timestep in episode
        self.epsilon = epsilon #determines how exploratory the b policy is
        self.q = q #value fn
        self.pi = pi #target policy
        self.reward_tracker = reward_tracker #stores accumulated rewards per episode
        self.state_tracker = state_tracker #Tracks how many times a state is visited
        self.mu = mu #mean of self.f
        self.sigma = sigma #standard deviation of self.f
        self.scale = scale #scale parameter for self.f
    
    def f(self,x):
        y = np.multiply(np.exp(np.divide(-1*np.power((x-self.mu),2),2*np.power(self.sigma,2))),np.divide(self.scale,2.50663*self.sigma))
        return y
        

    def init(self):
        rng = np.random.default_rng()    

        #initialize q for every state and action pair
        for z in np.arange(0,101,1):
            for slope in np.round(np.linspace(-60,60,12001),2):
                temp = {}
                for action in [-15,-7,-5,-4,-2,-1,1,2,4,5,7,15]:
                    if action + z < 100 and action + z > 0:
                        temp[action] = rng.integers(-15,15)
                self.q[(z,slope)] = temp
                self.state_tracker[(z,slope)] = 0

        #init pi to be greedy wrt q for every state action pair
        for key in self.q.keys():
            self.pi[key] = list(self.q[key].keys())[np.argmax(list(self.q[key].values()))]  
        self.data = {"s":[],"a":[],"r":[-1]}
        
    def terminal_check(self,t):
        
        slope = np.divide(np.subtract(self.f(self.data["s"][t+1][0]),self.f(self.data["s"][t][0])),np.abs(self.data["a"][t][0]))

        return slope <= self.delta and slope >= 0

    def get_slope(self,t):

        slope = np.divide(np.subtract(self.f(self.data["s"][t][0]+self.data["a"][t][0]),self.f(self.data["s"][t][0])),np.abs(self.data["a"][t][0]))

        return slope.round(2)

    def leaf_node_value(self,t):
        
        #Get average value of leaf nodes
        for action, val in self.q[self.data["s"][t]].items():
            if action == self.pi[self.data["s"][t]] and action != self.data["a"][t][0]:
                action_value = val
                action_chance =  1-self.epsilon+(self.epsilon/len(self.q[self.data["s"][t]]))
                return action_value*action_chance
        return 0
                
                
    def save(self,version):
        
        with open(f"gantryV2_q_{version}.pickle","wb") as f:
            pickle.dump(self.q,f)

        # #Write Value function to txt file
        # open(f"valueV2_fn{version}.txt","w").close()
        # with open(f"valueV2_fn{version}.txt","a") as f:
        #     f.write("========================================== Q ==========================================\n\n")
        # for state, actions in self.q.items():
        #     if self.state_tracker[state] != 0:
        #         with open(f"valueV2_fn{version}.txt","a") as f:
        #             f.write(f"----------> State: {state}\n")
        #             f.write(f"-----> Visits: {self.state_tracker[state]}\n\n")
        #         for action, value in actions.items():
        #             with open(f"valueV2_fn{version}.txt","a") as f:
        #                 f.write(f"Action|Value: {action} | {value}\n\n")
        

        # #Write Pi to txt file
        # open(f"piV2_{version}.txt","w").close()
        # with open(f"piV2_{version}.txt","a") as f:
        #    f.write("========================================== \u1d28 ==========================================\n\n")           
        # for state, action in self.pi.items():
        #    if self.state_tracker[state] != 0:            
        #        with open(f"piV2_{version}.txt","a") as f:
        #            f.write(f"state: {state} | action: {action}\n\n")                    

        print("Saved!")
        
        
    
    def bpolicy_action(self,state):

        #gets action according to behaviour policy
        #b policy is epsilon greedy wrt value fn[q] and target policy[pi]
        rng = np.random.default_rng()
        choice = rng.random()

        action_space = list(self.q[state].keys())
        greedy_action = self.pi[state]

        if choice >= self.epsilon:
            action_chance = 1-self.epsilon+(self.epsilon/len(action_space))
            action = greedy_action
        else:
            action_chance = 0
            action = rng.choice(action_space)

        arr = [action,action_chance]
        self.data["a"].append(arr)


    def step(self,t):

        #input: time step | output: new state
        self.data["r"].append(-1)
        action = self.data["a"][t][0]
        z = self.data["s"][t][0]
        new_state = (z+action,self.get_slope(t))
        self.state_tracker[new_state] += 1
        self.data["s"].append(new_state)

    def episode(self,i):
        print("**************************************************")
        print(f"starting episode {i+1}...\n")
        
        self.data = {"s":[],"a":[],"r":[-1]}        
        rng = np.random.default_rng()

        #initialize random start state & function and determine first action
        new_sigma = np.round(rng.normal(20,3),0)
        self.mu = np.round(rng.normal(50,12),0) 
        self.sigma = new_sigma if new_sigma != 0 else 10
        
        initial_z = rng.integers(0,101)
        # initial_slope = rng.choice(np.round(np.linspace(-60,60,12001),2))
        initial_state = (initial_z,0.01)
        
        self.state_tracker[initial_state] += 1
        self.data["s"].append(initial_state)

        print(f"initial (mu, sigma): ({self.mu}, {self.sigma})\n")
        print(f"initial state: {initial_state}")
        self.bpolicy_action(initial_state)

        T = np.inf #terminal time step
        t = 0 #current time step
        tau = 0 # time step n steps behind current time step
        while tau != T-1:

            #take action and check for terminal state, else determine next action
            if t < T:
                
                self.step(t)
                
                if self.terminal_check(t):
                    difference = np.abs(self.data["s"][t+1][0]-self.mu)
                    if difference > 10:
                        self.data["r"][t+1] = -100                        
                    elif difference != 0:
                        self.data["r"][t+1] = (1/difference)*100
                    else:
                        self.data["r"][t+1] = 200
                    print(f"final state: {self.data['s'][t+1]}")
                    print(f"\nfinished episode on step: {t+1}")
                    T = t+1
                    self.reward_tracker = np.append(self.reward_tracker,sum(list(self.data["r"])))
                    print(f"final cumulative reward: {self.reward_tracker[-1]}")
                else:
                    self.bpolicy_action(self.data["s"][t+1])

            #tau is the time step at which the value fn is being updated
            tau = t+1-self.n
            if tau >= 0:
                #determines the discounted update target based on if next time step is terminal
                if t+1 >= T:
                    returns = self.data["r"][T]
                else:
                    action_chance = 1-self.epsilon+(self.epsilon/len(self.q[self.data["s"][t+1]]))                
                    action_value = 0
                    for action, val in self.q[self.data["s"][t+1]].items():
                        if action == self.pi[self.data["s"][t+1]]:
                            action_value = val
                    returns = self.data["r"][t+1]+0.9*action_value*action_chance

                #updates value fn and target policy toward discounted update target[returns]
                for k in np.arange(min(t,T-1),tau+1,-1):
                    returns = self.data["r"][k] + 0.9 * self.leaf_node_value(k) + 0.9 * returns * self.data["a"][k][1]

                self.q[self.data["s"][tau]][self.data["a"][tau][0]] += self.alpha*(returns - self.q[self.data["s"][tau]][self.data["a"][tau][0]])
                self.pi[self.data["s"][tau]] = list(self.q[self.data["s"][tau]].keys())[np.argmax(list(self.q[self.data["s"][tau]].values()))]
                   
            if tau == T-1:
                print("**************************************************\n")
            t+=1
            

def main():
    m = control()
    m.init()

    numruns = int(1e6)
    for i in range(numruns):
        m.episode(i)

    m.save("1e6")

    
    i = 0
    avg_reward = np.array([])
    while i < numruns:
        avg_reward = np.append(avg_reward,np.mean(m.reward_tracker[i:i+1000]))
        i += 1000

    print(f"final average reward: {avg_reward[-1]}")
    fig, ax = plt.subplots()

    ax.set_xlabel("episode")
    ax.set_ylabel("avg reward")

    ax.plot(np.linspace(0,numruns,int(numruns/1000)),avg_reward,"-",color="royalblue")
    plt.savefig("reward_plot.png")
    plt.show()
    plt.close()
            

if __name__ == "__main__":
    main()
                                                    
                                                    
            
