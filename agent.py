import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'green'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

 
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        
        self.state = { 
                      'next_waypoint':self.next_waypoint,
                      'left':inputs['left'],
                      'oncoming':inputs['oncoming'],
                      'light':inputs['light'],
                     }
           
        #print "state = " + self.state 
        
        # TODO: Select action according to your policy
        action = random.choice(Environment.valid_actions[1:])

        # Execute action and get reward
        #self.env.sense(self)
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

#ref: https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/
#https://github.com/e-dorigatti/tictactoe
#https://gist.github.com/fheisler/430e70fa249ba30e707f

def choose_action(self, state):
    q = [self.getQ(state, a) for a in self.actions]
    maxQ = max(q)
 
    if  1:
        best = [i for i in range(len(self.actions)) if q[i] == maxQ]
        i = random.choice(best)
    else:
        i = q.index(maxQ)
 
    action = self.actions[i]
 
    return action
    

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(QLearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(QLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'green'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.QPlayer = QLearningPlayer(0.2,0.8,0.8)
        self.QPlayer.start_game('x')
 
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # slowly increase gamma or alpha 
        
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        
        self.state = { 
                      self.next_waypoint,
                      inputs['left'],
                      inputs['oncoming'],
                      inputs['light'],
                     }
                     
        #print self.state        
        
        action = random.choice(Environment.valid_actions[1:])
        action = self.QPlayer.move(self.state)
        # Execute action and get reward
        reward = self.env.act(self, action)
        #print action, reward
        self.env.sense(self) #sense the environment
        self.QPlayer.reward(reward,self.state)

        # TODO: Learn policy based on state, action, reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

#ref: https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/
#https://github.com/e-dorigatti/tictactoe
#https://gist.github.com/fheisler/430e70fa249ba30e707f


class QLearningPlayer():
    def __init__(self, epsilon=0.2, alpha=0.8, gamma=0.8):
        self.breed = "Qlearner"
        self.harm_humans = False
        self.q = {} # (state, action) keys: Q values
        self.epsilon = epsilon # e-greedy chance of random exploration
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor for future rewards

    def start_game(self, char):
        
        self.last_state = None
        self.last_move = None

    def getQ(self, state, action):
        # encourage exploration; "optimistic" 1.0 initial values
        if self.q.get((state, action)) is None:
            self.q[(state, action)] = 1.0
        #print self.q.get((state, action))    
        return self.q.get((state, action))

    def move(self,state):
        self.last_state = tuple(state)
        
        actions = Environment.valid_actions#self.available_moves(board)
        
                    
        if random.random() < self.epsilon: # explore!
            self.last_move = random.choice(Environment.valid_actions[1:])
            return self.last_move

        qs = [self.getQ(self.last_state,a) for a in actions]
        maxQ = max(qs)
        #print maxQ

        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(actions)) if qs[i] == maxQ]
            i = random.choice(best_options)
        else:
            i = qs.index(maxQ)

        self.last_move = actions[i]
        return actions[i]

    def reward(self, value, state):
        #print self.last_state
        if self.last_move:
           self.learn(self.last_state, self.last_move, value,tuple(state))
           

    def learn(self, state, action, reward,result_state):
        prev = self.getQ(state, action)
        maxqnew = max([self.getQ(result_state,a) for a in Environment.valid_actions])
                   #self.available_moves(state)])
        self.q[(state, action)] = prev + self.alpha * ((reward + self.gamma*maxqnew) - prev)

if __name__ == '__main__':
    run()
