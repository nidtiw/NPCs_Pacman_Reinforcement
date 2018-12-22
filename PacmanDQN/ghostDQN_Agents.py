# Used code from
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

# Used code from:
# The Pacman AI projects were developed at UC Berkeley found at
# http://ai.berkeley.edu/project_overview.html

import numpy as np
import random
import util
import time
import sys
# Pacman game
from pacman import Directions
from ghosts import GhostRules
from game import Agent
import game
# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf
from DQN import *

params = {
    # Model backups
    # intermediate ghost for bigger grid - not good
    # 'load_file': '/projectnb/dl-course/vidyaam/project/DL_RL_CollisionAvoidance/PacmanDQN/saves/model-ghosts_classic_199471_479',

    # 'load_file': '/projectnb/dl-course/nidhi/project/taken_swarnim/DL_RL_CollisionAvoidance/PacmanDQN/saves/ghosts_medium_more_iterations/model-ghosts_medium_more_iterations_398990_16961',
    # 'load_file': 'saves/model-ghost_medium_final_570356_17961',
    # 'save_file': 'ghost_medium_final',
    # 'load_file': 'saves/model-ghosts_two_against_medium_513960_16990',
    'ghosts_models': ['model-ghost1medium_classic_3_ghosts_50_1806160_18962', 'model-ghost2medium_classic_3_ghosts_50_1796200_18922', 'model-ghost3medium_classic_3_ghosts_50_1796172_18950'],
    # 'load_file': None,
    # 'save_file': 'swarnim_ghosts_before_11',
    # 'save_file': 'medium_classic_3_ghosts_50',
    'save_file': None,
    'save_interval' : 10000,

    # Training parameters
    'train_start': 5000,    # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 100000,     # Replay memory size

    'discount': 0.95,       # Discount rate (gamma value)
    'lr': .0002,            # Learning reate
    # 'rms_decay': 0.99,      # RMS Prop decay (switched to adam)
    # 'rms_eps': 1e-6,        # RMS Prop epsilon (switched to adam)

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value
    'eps_final': 0.1,       # Epsilon end value
    'eps_step': 10000       # Epsilon steps between start and end (linear)
}


class ghostDQN(Agent):
    def __init__(self, index):

        print("Initialise DQN Agent")

        # Load parameters from user-given arguments
        params['load_file'] = '/projectnb/dl-course/nidhi/project/taken_swarnim/DL_RL_CollisionAvoidance/PacmanDQN/saves/'+params['ghosts_models'][index-1]
        self.params = params
        self.params['width'] = 20
        self.params['height'] = 11
        self.params['num_training'] = 400
        # TODO: make this dynamic - for different ghosts
        self.index = index
        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        self.qnet = DQN(self.params)

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global = []
        self.cost_disp = 0

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.
        self.lastdist = 15

        self.replay_mem = deque()
        self.last_scores = deque()

        # self.lastdist = deque()


    def getMove(self, state):
        # Exploit / Explore
        if np.random.rand() > self.params['eps']:
            # Exploit action
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict = {self.qnet.x: np.reshape(self.current_state,
                                                     (1, self.params['width'], self.params['height'], 6)),
                             self.qnet.q_t: np.zeros(1),
                             self.qnet.actions: np.zeros((1, 4)),
                             self.qnet.terminals: np.zeros(1),
                             self.qnet.rewards: np.zeros(1)})[0]

            self.Q_global.append(max(self.Q_pred))
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))
            if len(a_winner) == 0 :
                move = self.getRandom(state)
    
            else:
                if len(a_winner) > 1:
                    # iterate over the winning moves
                    for i in range(len(a_winner)):
                        # for each move check if the current is leagal
                        move = self.get_direction(a_winner[i][0])
                        if self.isLegal(state, move):
                            # if the current move is legal, set it as the current move
                            # rather than random
                            break
                    # move = self.get_direction(
                    #     a_winner[np.random.randint(0, len(a_winner))][0])
                else:
                    move = self.get_direction(
                        a_winner[0][0])
        else:
            # Random:
            move = self.getRandom(state)
        # todo - check if the move set is legal for the ghost
        # if illegal, use the random move given in the ghostAgents.py
        if not self.isLegal(state, move):
            move = self.getRandom(state)
        # Save last_action
        self.last_action = self.get_value(move)

        return move

    def isLegal(self, state, move):
        """
        return true if the move returned by the DQN is a valid move for the ghost or not
        """
        possibleMoves = GhostRules.getLegalActions(state, self.index)
        return move in possibleMoves

    def getRandom(self, state):
        """
        return a randomly generated move if the move returned by DQN is invalid
        """
        possibleMoves = GhostRules.getLegalActions(state, self.index)
        move = possibleMoves[np.random.randint(0,len(possibleMoves))]
        return move

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST

    def getOtherGhostMatrix(self, state, index):
        """ Return a matrix for the 1st ghost's position set to 1 """
        width, height = state.data.layout.width, state.data.layout.height
        matrix = np.zeros((height, width), dtype=np.int8)
        # mark the index of the 1st ghost as 1
        pos = state.data.agentStates[index].configuration.getPosition()
        cell = 1
        matrix[-1-int(pos[1])][int(pos[0])] = cell
        
        return matrix
    def getPacmanMatrix(self,state):
        """ Return matrix with pacman coordinates set to 1 """
        width, height = state.data.layout.width, state.data.layout.height
        matrix = np.zeros((height, width), dtype=np.int8)

        for agentState in state.data.agentStates:
            if agentState.isPacman:
                pos = agentState.configuration.getPosition()
                cell = 1
                matrix[-1-int(pos[1])][int(pos[0])] = cell

        return matrix

    def getGhostMatrix(self,state):
        """ Return matrix with ghost coordinates set to 1 """
        width, height = state.data.layout.width, state.data.layout.height
        matrix = np.zeros((height, width), dtype=np.int8)

        for agentState in state.data.agentStates:
            if not agentState.isPacman:
                if not agentState.scaredTimer > 0:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

        return matrix

    def observation_step(self, state):
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(state)

            self.previous_ghost_matrix = np.copy(self.current_ghost_matrix)
            self.current_ghost_matrix = self.getPrevGhostStateMatrices(state)
            

            # Process current experience reward
            self.current_score = state.getScore()
            
            reward = self.current_score - self.last_score

            self.last_score = self.current_score


            pac_state = self.getPacmanMatrix(state)
            ghost_state = self.getGhostMatrix(state)

            if len(np.where(ghost_state ==1)[0])==0:
                dist = self.lastdist
            else:
                dist = self.findManhattanDistance(pac_state,ghost_state)
            self.current_dist = dist
            movement = self.lastdist -  self.current_dist 
        
            if reward > 20:
                # pacman ate the ghost - punish heavily
                self.last_reward = -150.    # Eat ghost   (Yum! Yum!)
            elif reward > 0:
                self.last_reward = -10.    # Eat food    (Yum!)
            elif reward < -10:
                self.last_reward = 500.  # Get eaten   (Ouch!) -500
                self.won = True

            elif movement < 0:
                # moving away = -1
                self.last_reward = -1
            # todo create reward system for ghost 2
            # if the x or y co-ordinates for both of the ghosts and the pacman
            # is the same, then reward mirror movements
            # else punish mirror movements
            width, height = state.data.layout.width, state.data.layout.height


            if self.index >= 2:
                prev_other_ghost_matrix = self.previous_ghost_matrix[0]
                current_other_ghost_matrix = self.getOtherGhostMatrix(state, 1)

                prev_responder_ghost_matrix = self.previous_ghost_matrix[self.index-1]
                current_responder_ghost_matrix = self.getOtherGhostMatrix(state, self.index)
                
                x_prev_ghost_other = np.where(prev_other_ghost_matrix ==1)[0][0]
                y_prev_ghost_other = np.where(prev_other_ghost_matrix ==1)[1][0]

                x_curr_ghost_other = np.where(current_other_ghost_matrix ==1)[0][0]
                y_curr_ghost_other = np.where(current_other_ghost_matrix ==1)[1][0]

                x_prev_ghost_responder = np.where(prev_responder_ghost_matrix ==1)[0][0]
                y_prev_ghost_responder = np.where(prev_responder_ghost_matrix ==1)[1][0]

                x_curr_ghost_responder = np.where(current_responder_ghost_matrix ==1)[0][0]
                y_curr_ghost_responder = np.where(current_responder_ghost_matrix ==1)[1][0]

                x_pac = np.where(pac_state == 1)[0][0]
                y_pac = np.where(pac_state == 1)[1][0]

                same_x = False
                same_y = False

                if x_prev_ghost_responder == x_curr_ghost_responder == x_pac:
                    same_x = True

                if y_prev_ghost_responder == y_curr_ghost_responder == y_pac:
                    same_y = True

                if same_x:
                    if np.sign(y_prev_ghost_other - y_curr_ghost_other) != np.sign(y_prev_ghost_responder - y_prev_ghost_responder):
                        self.last_reward += 10
                    else:
                        self.last_reward -= 1
                elif same_y:
                    if np.sign(x_prev_ghost_other - x_curr_ghost_other) != np.sign(x_prev_ghost_responder - x_prev_ghost_responder):
                        self.last_reward += 10
                    else:
                        self.last_reward -= 1



                # penalize mirroring for normal cases when they don't lie on the same axes
                elif (x_prev_ghost_other - x_curr_ghost_other) * (x_prev_ghost_responder - x_prev_ghost_responder) < 0:
                    self.last_reward -= 5

                elif (y_prev_ghost_other - y_curr_ghost_other) * (y_prev_ghost_responder - y_prev_ghost_responder) < 0:
                    self.last_reward -= 5

                else:
                    self.last_reward += 1
            
            self.lastdist = self.current_dist

            if(self.terminal and self.won):
                self.last_reward += 100.
            self.ep_rew += self.last_reward

            # Store last experience into memory
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            if(params['save_file']):
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                    self.qnet.save_ckpt('saves/model-ghost'+str(self.index) + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt)/ float(self.params['eps_step']))

    def findManhattanDistance(self,pacMat,ghostMat):
        
        pac_x = np.where(pacMat ==1)[0][0]
        pac_y = np.where(pacMat ==1)[1][0]
        #print(np.where(ghostMat==1))
        ghost_x = np.where(ghostMat ==1)[0][0]
        ghost_y = np.where(ghostMat ==1)[1][0]
        dist = np.abs(pac_x - ghost_x) + np.abs(pac_y - ghost_y)
        
        return dist

    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Print stats
        log_file = open('./logs/ghost-'+str(self.index)+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(self.params['height'])+'-x-'+str(self.params['num_training'])+'.log','a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()

    def train(self):
        # Train
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = [] # States (s)
            batch_r = [] # Rewards (r)
            batch_a = [] # Actions (a)
            batch_n = [] # Next states (s')
            batch_t = [] # Terminal state (t)

            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)


    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):
            actions_onehot[i][int(actions[i])] = 1
        return actions_onehot

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getPrevGhostStateMatrices(self, state):
        width, height = state.data.layout.width, state.data.layout.height
        matrix1 = np.zeros((height, width), dtype=np.int8)

        
        pos = state.data.agentStates[1].configuration.getPosition()
        cell = 1
        matrix1[-1-int(pos[1])][int(pos[0])] = cell

        matrix2 = np.zeros((height, width), dtype=np.int8)
        matrix3 = np.zeros((height, width), dtype=np.int8)

        if len(state.data.agentStates) > 2:
            pos = state.data.agentStates[2].configuration.getPosition()
            cell = 1
            matrix2[-1-int(pos[1])][int(pos[0])] = cell

            if len(state.data.agentStates) > 3:
                pos = state.data.agentStates[3].configuration.getPosition()
                cell = 1
                matrix3[-1-int(pos[1])][int(pos[0])] = cell

        return matrix1, matrix2, matrix3

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return observation

    def registerInitialState(self, state): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # reset ghost matrix
        self.current_ghost_matrix = self.getPrevGhostStateMatrices(state)
        self.previous_ghost_matrix = None

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = False
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        # Stop moving when not legal
        legal = state.getLegalActions(self.index)
        if move not in legal:
            move = self.getRandom(state)
        return move
