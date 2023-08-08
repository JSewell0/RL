import numpy as np
import matplotlib.pyplot as plt
import pickle


with open("gantryV2_q_1e6.pickle", "rb") as f:
    value_fn = pickle.load(f)

class control:
    def __init__(self, alpha=0.05, n=5, delta=0.025,data={},epsilon=0.1, q={}, pi={},mu=40,sigma=20,scale=1000):
        self.alpha = alpha
        self.n = n
        self.delta = delta
        self.data = data
        self.epsilon = epsilon
        self.q = q
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.scale = scale        
    
    def f(self,x,scale=1000):
        y = np.multiply(np.exp(np.divide(-1*np.power((x-self.mu),2),2*np.power(self.sigma,2))),np.divide(self.scale,2.50663*self.sigma))
        return y
        

    def init(self):
        self.q = value_fn

        # for key in self.q.keys():
        #     self.pi[key] = list(self.q[key].keys())[np.argmax(list(self.q[key].values()))]        
        

    def terminal_check(self,t):
        
        slope = np.divide(np.subtract(self.f(self.data["s"][t+1][0]),self.f(self.data["s"][t][0])),np.abs(self.data["a"][t]))

        return slope <= self.delta and slope >= 0

    def leaf_node_value(self,t):
        for action, val in self.q[self.data["s"][t]].items():
            if action == self.pi[self.data["s"][t]] and action != self.data["a"][t][0]:
                action_value = val
                action_chance =  1-self.epsilon+(self.epsilon/len(self.q[self.data["s"][t]]))
                return action_value*action_chance
        return 0

    def visualize(self,t):
        z = np.arange(0,101,1)
        plt.plot(z,self.f(z),"-",color="royalblue")#plot function
        plt.plot(self.mu,self.f(self.mu),marker="X",mec="r",mfc="r")
        plt.plot(self.data['s'][t][0],self.f(self.data['s'][t][0]),marker="s",mec="k",mfc="k")#plot current state
        plt.show()
        plt.close()
                
    def save(self):
        with open("gantry_q_pi5e3v2.pickle","wb") as f:
            pickle.dump((self.q,self.pi),f)

    def get_slope(self,t):

        slope = np.divide(np.subtract(self.f(self.data["s"][t][0]+self.data["a"][t]),self.f(self.data["s"][t][0])),np.abs(self.data["a"][t]))

        return slope.round(2)            
        
    
    def bpolicy_action(self,state):
        
        action = list(self.q[state].keys())[np.argmax(list(self.q[state].values()))]  
        print(f"action: {action}\n")
        self.data["a"].append(action)

        
    def step(self,t):

        #input: time step | output: new state
        self.data["r"].append(-1)
        action = self.data["a"][t]
        z = self.data["s"][t][0]
        new_state = (z+action,self.get_slope(t))
        print(f"new state: {new_state}")
        self.data["s"].append(new_state)

    def episode(self):
        
        print("**************************************************")
        self.data = {"s":[],"a":[],"r":[-1]}        
        rng = np.random.default_rng()
        
        new_sigma = np.round(rng.normal(20,3),0)
        self.mu = np.round(rng.normal(50,15),0) 
        self.sigma = new_sigma if new_sigma != 0 else 10
        
        initial_z = rng.integers(0,101)
        initial_slope = rng.choice(np.round(np.linspace(-60,60,12001),2))
        initial_state = (initial_z,initial_slope)
        
        self.data["s"].append(initial_state)        
        print(f"initial state: {initial_state}\n")
        print(f"intial mu,sigma: ({self.mu}, {self.sigma})\n")
        self.bpolicy_action(initial_state)
        
        t = 0
        tau = 0
        running = True
        while running:
            self.visualize(t)

            if t == 100:
                print("Did not find minimum")
                break
            self.step(t)
            
            if self.terminal_check(t):
                self.data["r"][t+1] = self.data["s"][t][1]*0.9
                self.bpolicy_action(self.data["s"][t+1])                
                print(f"final state: {self.data['s'][t+1]}\n")
                print(f"finished episode on step: {t+1}")
                print("**************************************************")
                self.visualize(t+1)                
                running = False
            else:
                self.bpolicy_action(self.data["s"][t+1])
                t+=1
                    

            
m = control()
m.init()
        
m.episode()

