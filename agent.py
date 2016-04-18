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
        self.QPlayer = QLearningPlayer()
 
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
        action_okay = True
        if self.next_waypoint == 'right':
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif self.next_waypoint == 'forward':
            if inputs['light'] == 'red':
                action_okay = False
        elif self.next_waypoint == 'left':
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                action_okay = False

        action = None
        if action_okay:
            action = self.next_waypoint
            #self.next_waypoint = random.choice(Environment.valid_actions[1:])
        #reward = self.env.act(self, action)

        qp = QLearningPlayer()
        qp.start_game(qp)
        action = qp.move(self.state)
        
        #print action
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        qp.reward(reward,self.state)


        # TODO: Learn policy based on state, action, reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

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
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=.1)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit

class QLearningPlayer():
    def __init__(self, epsilon=0.2, alpha=0.3, gamma=0.9):
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
        return self.q.get((state, action))

    def move(self,state):
        self.last_state = tuple(state)
        actions = Environment.valid_actions#self.available_moves(board)
        
                    
        if random.random() < self.epsilon: # explore!
            self.last_move = random.choice(Environment.valid_actions[1:])
            return self.last_move

        qs = [self.getQ(self.last_state,a) for a in actions]
        maxQ = max(qs)

        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(actions)) if qs[i] == maxQ]
            i = random.choice(best_options)
        else:
            i = qs.index(maxQ)

        self.last_move = actions[i]
        return actions[i]

    def reward(self, value, state):
        if self.last_move:
           self.learn(self.last_state, self.last_move, value,tuple(state))

    def learn(self, state, action, reward,result_state):
        prev = self.getQ(state, action)
        maxqnew = max([self.getQ(result_state,a) for a in Environment.valid_actions])
                   #self.available_moves(state)])
        self.q[(state, action)] = prev + self.alpha * ((reward + self.gamma*maxqnew) - prev)

if __name__ == '__main__':
    run()
