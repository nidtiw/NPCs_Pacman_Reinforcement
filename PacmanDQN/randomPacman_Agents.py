# randomPacman.py
# for the random movements of the pacman

from game import Agent
from game import Directions
import random

class RandomPacman(Agent):

    def __init__(self, index=0):

        self.lastMove = Directions.STOP
        self.keys = []

    def getAction(self, state):

        legal = state.getLegalActions(0)
        move = random.choice(legal)

        self.lastMove = move
        return move
