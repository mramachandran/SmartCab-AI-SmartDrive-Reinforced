import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.epsilon=0.2
        self.alpha=0.8
        self.gamma=0.8
        self.q = {} # (state, action) keys: Q values
        
        #placeholders for post processing the results
        self.resultAnalysis = {}
        self.resultAnalysis2 = {}
        self.numOfTrials = 0
        self.netReward = 0
        self.noOfStepsToDestination = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.start_game('x')
        # TODO: Prepare for a new trip; reset any variables here, if required
        #if self.numOfTrials == 10:
            
          #  self.analyzeResult()


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
                      inputs['light']
                        
                     }
        
        # TODO: Select action according to your policy
        action = self.move(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        self.updateQ(reward,self.state)  
        # TODO: Learn policy based on state, action, reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
    def start_game(self, char):
        
        self.last_state = None
        self.last_move = None
        self.resultAnalysis[(self.numOfTrials)] = self.noOfStepsToDestination
        self.resultAnalysis2[(self.numOfTrials)] = self.netReward
        
        self.numOfTrials+= 1
        self.netReward = 0
        self.noOfStepsToDestination  = 0

    def getQ(self, state, action):
        # encourage exploration; "optimistic" 1.0 initial values
        if self.q.get((state, action)) is None:
            self.q[(state, action)] = 1.0
        #print self.q.get((state, action))    
        return self.q.get((state, action))

    def move(self,state):
        self.last_state = tuple(state)
        
        actions = Environment.valid_actions[1:]#self.available_moves(board)
        
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

    def updateQ(self, value, state):
        #print self.last_state
        if self.last_move:
           self.learn(self.last_state, self.last_move, value,tuple(state))
           #track netreward and total number of steps
           self.netReward += value
           self.noOfStepsToDestination += 1

    def learn(self, state, action, reward,result_state):
        prev = self.getQ(state, action)
        maxqnew = max([self.getQ(result_state,a) for a in Environment.valid_actions])
        #update the previous state
        self.q[(state, action)] = prev + self.alpha * ((reward + self.gamma*maxqnew) - prev)

    def analyzeResult(self):
        
        self.fig = plt.figure(figsize=(8,5))
        x = np.arange(len(self.resultAnalysis))
        y = self.resultAnalysis.values()
        
        print "Average Number of Steps to Goal, inputs = {}".format(np.mean(y))  # [debug]
        plt.plot(x,y , 'ro-', linewidth=2)
        plt.title('Learning Performance Plot')
        plt.xlabel('No. of trials')
        plt.ylabel('No. of steps to goal')
 
        leg = plt.legend(['Iterations to reach goal'], loc='best', borderpad=0.3,
                         shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                         markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)
        plt.show()
        
        self.fig2 = plt.figure(figsize=(8,5))
        x = np.arange(len(self.resultAnalysis2))
        y = self.resultAnalysis2.values()
        
        print "Average Net Reward, inputs = {}".format(np.mean(y)) 
        #print y
        plt.plot(x,y , 'ro-', linewidth=2)
        plt.title('Learning Performance Plot')
        plt.xlabel('No. of trials')
        plt.ylabel('Net Reward')
        #I don't like the default legend so I typically make mine like below, e.g.
        #with smaller fonts and a bit transparent so I do not cover up data, and make
        #it moveable by the viewer in case upper-right is a bad place for it
        leg = plt.legend(['Net Reward'], loc='best', borderpad=0.3,
                         shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                         markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)
        plt.show()  
        
        
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.1)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit

 

if __name__ == '__main__':
    run()
