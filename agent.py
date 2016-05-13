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
        self.color = 'green'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.breed = "RandomLearner"
 
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
        #Our policy in this scenario is to execute a random action
        action = random.choice(Environment.valid_actions[1:])

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

#ref: https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/
#https://github.com/e-dorigatti/tictactoe
#https://gist.github.com/fheisler/430e70fa249ba30e707f

AvgReward = {}
AvgSteps  = {}    

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    #qla = QLearningAgent(e)       
 
     
    #for learningrate in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9,1.0]:
    for learningrate in [0.5]:
        
        a = e.create_agent(QLearningAgent,learningrate)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
    
        # Now simulate it
        sim = Simulator(e, update_delay=0.010)  # reduce update_delay to speed up simulation
        sim.run(n_trials=100)  # press Esc or close pygame window to quit
    
    plotStudyGraph()      

class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env,LearningRate):
        super(QLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'green'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here        
        #improvement :
        #update epsilon to decrease to make  more non-random decisions as we
        #gather more Q-value for states
        self.deadline = 100
        self.noOfStepsToDestination = 0
        
        self.epsilon = 0.9*(1-(self.noOfStepsToDestination/self.deadline))  
        print self.epsilon                  
                 
        self.QPlayer = QLearningPlayer(0.2,LearningRate,0.8)
        self.QPlayer.start_game('x')
        self.netReward = 0
        
        self.resultAnalysis = {}
        self.resultAnalysis2 = {}
        self.numOfTrials = 0
        self.breed = "QLearner"
        self.LearningRate = LearningRate
        
 
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # slowly increase gamma or alpha 
        
        #print self.netReward
        self.resultAnalysis[(self.numOfTrials)] = self.noOfStepsToDestination
        self.resultAnalysis2[(self.numOfTrials)] = self.netReward
        
        
        
        self.netReward = 0
        self.noOfStepsToDestination = 0
        #print self.resultAnalysis        
        
        if self.numOfTrials == 99:
            #print self.resultAnalysis
            self.analyzeResult()
            AvgReward[self.LearningRate] = np.mean(self.resultAnalysis.values())
            AvgSteps[self.LearningRate]  = np.mean(self.resultAnalysis2.values())
            
        self.numOfTrials += 1    
        

                       
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        self.deadline = self.env.get_deadline(self)

        # TODO: Update state
        
        self.state = { 
                      self.next_waypoint,
                      inputs['left'],
                      inputs['oncoming'],
                      inputs['light'],
                     }
                     
        #print self.state        
        
        #action = random.choice(Environment.valid_actions[1:])
        action = self.QPlayer.move(self.state)
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        #update netreward
        self.netReward += reward
        self.noOfStepsToDestination += 1
        
        #print action, reward
        self.env.sense(self) #sense the environment #not sure if this makes any difference
        self.QPlayer.reward(reward,self.state)
        
        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(self.deadline, inputs, action, reward)  # [debug]
        #print self.netReward   
        
        
        
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
        
        #print('Percent of trials that successfully completed: ' + "".format())
        #
         # [debug]

def plotStudyGraph():
                        
            fig = plt.figure(figsize=(8,5))
            #x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9,1.0]
            x = [0.5]
            y = AvgReward.values()
            print x
            print y
            print "Average Number of Steps to Goal, inputs = {}".format(np.mean(y))  # [debug]
            plt.plot(x,y , 'ro-', linewidth=1)
            plt.title('Learning Performance Plot')
            plt.xlabel('Alpha - Learning Rate')
            plt.ylabel('No. of steps to goal')
     
            leg = plt.legend(['Iterations to reach goal'], loc='best', borderpad=0.3,
                             shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                             markerscale=0.4)
            leg.get_frame().set_alpha(0.4)
            leg.draggable(state=True)
            plt.show()
            
            fig2 = plt.figure(figsize=(8,5))
            #x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9,1.0]
            x = [0.5]
            y = AvgSteps.values()
            print x 
            print y
            print "Average Net Reward, inputs = {}".format(np.mean(y)) 
            #print y
            plt.plot(x,y , 'ro-', linewidth=2)
            plt.title('Learning Performance Plot')
            plt.xlabel('Alpha - Learning Rate')
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
            
            #print('Percent of trials that successfully completed: ' + "".format())
            #
             # [debug]

#ref: https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/
#https://github.com/e-dorigatti/tictactoe
#https://gist.github.com/fheisler/430e70fa249ba30e707f


class QLearningPlayer():
    def __init__(self, epsilon=0.2, alpha=0.8, gamma=0.8):
        
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

    def reward(self, value, state):
        #print self.last_state
        if self.last_move:
           self.learn(self.last_state, self.last_move, value,tuple(state))
           

    def learn(self, state, action, reward,result_state):
        prev = self.getQ(state, action)
        maxqnew = max([self.getQ(result_state,a) for a in Environment.valid_actions])
        #update the previous state
        self.q[(state, action)] = prev + self.alpha * ((reward + self.gamma*maxqnew) - prev)


 


if __name__ == '__main__':
    run()
