import numpy as np
import matplotlib.pyplot as plt
import pickle

class control:
    def __init__(self, alpha=0.3, n=3, delta=0.3,data={},epsilon=0.1, q={}, pi={}):
        self.alpha = alpha
        self.n = n
        self.delta = delta
        self.data = data
        self.epsilon = epsilon
        self.q = q
        self.pi = pi
    
    def f(self,x,mu=40,width=1000,scale=20):
        y = np.multiply(np.exp(np.divide(-1*np.power((x-mu),2),width)),scale)
        return y
        

    def init(self):
        rng = np.random.default_rng()    

        #initialize q 
        for z in np.arange(0,101,1):
            temp = {}
            for action in [-10,-5,5,10]:
                if action + z < 100 and action + z > 0:
                    temp[action] = rng.integers(0,15)
            self.q[(z,self.f(z))] = temp

        #init pi to be greedy wrt q
        self.pi = {(i,self.f(i)): list(self.q[(i,self.f(i))].keys())[np.argmax(list(self.q[(i,self.f(i))].values()))] for i in np.arange(0,101,1)}
        self.data = {"s":[],"a":[],"r":[-1]}
        
    def terminal_check(self,t):
        
        slope = np.divide(np.subtract(self.data["s"][t+1][1],self.data["s"][t][1]),np.abs(self.data["a"][t][0]))

        return slope <= self.delta and slope >= 0

    def leaf_node_value(self,t):
        for action, val in self.q[self.data["s"][t]].items():
            if action == self.pi[self.data["s"][t]] and action != self.data["a"][t][0]:
                action_value = val
                action_chance =  1-self.epsilon+(self.epsilon/len(self.q[self.data["s"][t]]))
                return action_value*action_chance
        return 0
                
                
    def save(self,version):
        
        with open(f"gantry_q_pi_{version}.pickle","wb") as f:
            pickle.dump((self.q,self.pi),f)

        #Write Value function to txt file
        open(f"value_fn{version}.txt","w").close()
        with open(f"value_fn{version}.txt","a") as f:
            f.write("========================================== Q ==========================================\n\n")
        for state, actions in self.q.items():
            with open(f"value_fn{version}.txt","a") as f:
                f.write(f"----------> State: {state}\n")    
            for action, value in actions.items():
                with open(f"value_fn{version}.txt","a") as f:
                    f.write(f"Action|Value: {action}|{value}\n\n")
        

        #Write Pi to txt file
        open(f"pi{version}.txt","w").close()
        with open(f"pi{version}.txt","a") as f:
           f.write("========================================== \u1d28 ==========================================\n\n"
        for state, action in self.pi.items():
           with open(f"pi{version}.txt","a") as f:
               f.write(f"state: {state}|action: {action}\n\n")                    

        print("Saved!")
        
        
    
    def bpolicy_action(self,state):
        
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
        # print(f"action: {action}\n")
        self.data["a"].append(arr)


    def step(self,action,t):

        self.data["r"].append(-1)
        z = self.data["s"][t][0]
        new_state = (z+action,self.f(z+action))
        # print(f"state: {new_state}")        
        self.data["s"].append(new_state)

    def episode(self,i):
        print("**************************************************")
        print(f"starting episode {i}...")

        self.data = {"s":[],"a":[],"r":[-1]}        
        rng = np.random.default_rng()
        
        initial_z = rng.integers(0,101)
        initial_state = (initial_z, self.f(initial_z))
        self.data["s"].append(initial_state)        
        print(f"initial state: {initial_state}")
        self.bpolicy_action(initial_state)

        T = np.inf
        t = 0
        tau = 0
        while tau != T-1:
            
            if t < T:
                self.step(self.data["a"][t][0],t)
                if self.terminal_check(t):
                    self.data["r"][t+1] = self.data["s"][t][1]*1.5
                    print(f"final state: {self.data['s'][t+1]}")
                    print(f"finished episode on step: {t+1}")
                    T = t+1
                else:
                    self.bpolicy_action(self.data["s"][t+1])
                    
            tau = t+1-self.n
            if tau >= 0:
                if t+1 >= T:
                    returns = self.data["r"][T]
                else:
                    action_chance = 1-self.epsilon+(self.epsilon/len(self.q[self.data["s"][t+1]]))                
                    action_value = 0
                    for action, val in self.q[self.data["s"][t+1]].items():
                        if action == self.pi[self.data["s"][t+1]]:
                            action_value = val
                    returns = self.data["r"][t+1]+0.9*action_value*action_chance
                                                    
                for k in np.arange(min(t,T-1),tau+1,-1):
                    returns = self.data["r"][k] + 0.9 * self.leaf_node_value(k) + 0.9 * returns * self.data["a"][k][1]

                self.q[self.data["s"][tau]][self.data["a"][tau][0]] += self.alpha*(returns - self.q[self.data["s"][tau]][self.data["a"][tau][0]])
                self.pi[self.data["s"][tau]] = list(self.q[self.data["s"][tau]].keys())[np.argmax(list(self.q[self.data["s"][tau]].values()))]
            if tau == T-1:
                print("**************************************************\n")
            t+=1
            


m = control()
m.init()
for i in range(100000):
    m.episode(i)

m.save("1e5v3")
            

                
                                                    
                                                    
            
