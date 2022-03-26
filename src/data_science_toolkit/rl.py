import gym
import numpy as np

class Environment:
    
    def __init__(self, environment_name="FrozenLake-v0", is_slippery=True):
        """The environment constructor"""
        if environment_name == "FrozenLake-v0":
            self.__environment = gym.make(environment_name, is_slippery=is_slippery)
        else:
            self.__environment = gym.make(environment_name)
        self.__status = self.reset()
        
        
    def describe(self):
        print("Environement states: ", self.__environment.observation_space)
        print("Availble actions: ", self.__environment.action_space)
        # env.P[state][action] to get one transition probability
        # The result is in the form of: [(transition probability, next state, reward, Is terminal state?)]
        print(gym.spaces)
        print(isinstance(self.__environment.observation_space, gym.spaces.box.Box))
        #print("Transition probabilities matrix ", self.__environment.P)
        
    def get_transition_probability_state_action(self, state, action):
        """[(transition probability, next state, reward, Is terminal state?)]"""
        return self.__environment.P[state][action]
        
    def render(self):
        """show the environment"""
        self.__environment.render()
        
    def reset(self):
        """resetting puts our agent back to the initial state"""
        self.__status = self.__environment.reset()
        
    def close(self):
        """resetting puts our agent back to the initial state"""
        self.__environment.close()
        
    def step(self, action=None):
        """ step
        
        take a given action
        The step function return : (next_state, reward, done, info)
        if no action, a random action will be sampled
        """
        if action is None:
            action = self.__environment.action_space.sample()    
        self.__status = self.__environment.step(action)
    
    def status(self):
        """ The status is in the form : (next_state, reward, done, info) """
        return self.__status
    
    def is_terminal_state(self):
        """ check if the agent is in a terminal state"""
        if self.__status[2]:
            return True
        return False 
    
    def random_episode(self, number_of_times_steps=10):
        """random trajectory from an initial state to a final state possible only in an episodic 
        environement and not continous one
        """
        t = 0
        print('Time Step 0 :')
        self.render()
        self.step()
        while not self.is_terminal_state():
            self.render()
            print ('Time Step {} :'.format(t+1))
            self.step()
            t += 1
            
    def random_horizion(self, number_of_times_steps=10):
        """random trajectory starting from an initial state to given number of time steps
        """
        for i in range(number_of_times_steps):
            self.render()
            self.step()
            
    def all_environments(self):
        print(gym.envs.registry.all())
        
    def action_space_cardinal(self):
        return self.__environment.action_space.n
    
    def state_space_cardinal(self):
        return self.__environment.observation_space.n
    
    def value_iteration(self, num_iterations=1000, threshold=1e-20, gamma=1.0):
        value_table = np.zeros(self.state_space_cardinal())
        
        for i in range(num_iterations):
            updated_value_table = np.copy(value_table) 
            for s in range(self.state_space_cardinal()):
                Q_values = [sum([prob*(r + gamma * updated_value_table[s_])
                                for prob, s_, r, _ in self.get_transition_probability_state_action(s,a)])
                                    for a in range(self.action_space_cardinal())]
                                            
                value_table[s] = max(Q_values)
            
            if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
                break
        
        return value_table
    
    def extract_policy(self, value_table, gamma = 1.0):
        policy = np.zeros(self.state_space_cardinal()) 
    
        for s in range(self.state_space_cardinal()):
            
            Q_values = [sum([prob*(r + gamma * value_table[s_])
                                for prob, s_, r, _ in self.get_transition_probability_state_action(s, a)]) 
                                    for a in range(self.action_space_cardinal())]
                    
            policy[s] = np.argmax(np.array(Q_values)) 
        
        return policy
    
    
    def run_policy(self, policy):
        self.reset()
        state = self.status()
        print(policy)
        for p in range(50):
            self.step(int(policy[state]))
            state = self.status()[0]
            self.render()

