import numpy as np
import matplotlib.pyplot as plt
import pickle


with open("gantry_q_pi_1e5v3.pickle", "rb") as f:
    value_fn, pi = pickle.load(f)

class control:
    def __init__(self, alpha=0.3, n=6, delta=0.17,data={},epsilon=0.1, q={}, pi={}):
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
        self.q = value_fn
        self.pi = {(i,self.f(i)): list(self.q[(i,self.f(i))].keys())[np.argmax(list(self.q[(i,self.f(i))].values()))] for i in np.arange(0,101,1)}
        

        
    def terminal_check(self,t):
        
        slope = np.divide(np.subtract(self.data["s"][t+1][1],self.data["s"][t][1]),np.abs(self.data["a"][t]))

        return slope <= self.delta and slope >= 0

    def leaf_node_value(self,t):
        for action, val in self.q[self.data["s"][t]].items():
            if action == self.pi[self.data["s"][t]] and action != self.data["a"][t][0]:
                action_value = val
                action_chance =  1-self.epsilon+(self.epsilon/len(self.q[self.data["s"][t]]))
                return action_value*action_chance
        return 0
                
                
    def save(self):
        with open("gantry_q_pi5e3v2.pickle","wb") as f:
            pickle.dump((self.q,self.pi),f)
            
        
        
    
    def bpolicy_action(self,state):
        
        action = self.pi[state]
        print(f"action: {action}\n")
        self.data["a"].append(action)


    def step(self,action,t):

        self.data["r"].append(-1)
        z = self.data["s"][t][0]
        new_state = (z+action,self.f(z+action))
        print(f"state: {new_state}")
        self.data["s"].append(new_state)

    def episode(self):
        
        print("**************************************************")
        self.data = {"s":[],"a":[],"r":[-1]}        
        rng = np.random.default_rng()
        
        initial_z = rng.integers(0,101)
        initial_state = (initial_z, self.f(initial_z))
        self.data["s"].append(initial_state)        
        print(f"initial state: {initial_state}")
        self.bpolicy_action(initial_state)
        
        t = 0
        tau = 0
        running = True
        while running:

            if t == 50:
                break
            self.step(self.data["a"][t],t)
            
            if self.terminal_check(t):
                self.data["r"][t+1] = self.data["s"][t][1]*0.9
                print(f"\nfinal state: {self.data['s'][t+1]}")
                print(f"finished episode on step: {t+1}")
                print("**************************************************")                
                running = False
            else:
                self.bpolicy_action(self.data["s"][t+1])
                t+=1
                    

            
m = control()
m.init()
        
m.episode()

