import util 
from util import Belief, pdf 
from engine.const import Const
from math import sin,cos,sqrt
import random
import time

# Class: Estimator
#----------------------
# Maintain and update a belief distribution over the probability of a car being in a tile.
class Estimator(object):
    def __init__(self, numRows: int, numCols: int):
        self.belief = util.Belief(numRows, numCols) 
        self.transProb = util.loadTransProb() 
            
    ##################################################################################
    # [ Estimation Problem ]
    # Function: estimate (update the belief about a StdCar based on its observedDist)
    # ----------------------
    # Takes |self.belief| -- an object of class Belief, defined in util.py --
    # and updates it *inplace* based onthe distance observation and your current position.
    #
    # - posX: x location of AutoCar 
    # - posY: y location of AutoCar 
    # - observedDist: current observed distance of the StdCar 
    # - isParked: indicates whether the StdCar is parked or moving. 
    #             If True then the StdCar remains parked at its initial position forever.
    # 
    # Notes:
    # - Carefully understand and make use of the utilities provided in util.py !
    # - Remember that although we have a grid environment but \
    #   the given AutoCar position (posX, posY) is absolute (pixel location in simulator window).
    #   You might need to map these positions to the nearest grid cell. See util.py for relevant methods.
    # - Use util.pdf to get the probability density corresponding to the observedDist.
    # - Note that the probability density need not lie in [0, 1] but that's fine, 
    #   you can use it as probability for this part without harm :)
    # - Do normalize self.belief after updating !!

    ###################################################################################
    def estimate(self, posX: float, posY: float, observedDist: float, isParked: bool) -> None:
        # BEGIN_YOUR_CODE
        begin = time.time()
        numRows = self.belief.numRows
        numCols = self.belief.numCols

        # print(self.transProb)        
        std = Const.SONAR_STD
        trans_prob = self.transProb
        
        #setting the belief according to particle filter

        #N = numRows*numCols*12
        N = 10000
        beliefGrid = self.belief.grid

        particles = []

        for row in range(numRows):
            for col in range(numCols):
                number = round(beliefGrid[row][col]*N)
                for _ in range(number):
                    particles.append((row,col))

        new_particles = []

        for (x,y) in particles:
            position = (x,y)
            adj = [(x,y+1),(x,y-1),(x+1,y),(x-1,y),(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1),(x,y)]
            probs = []
            for neighbour in adj:
                if (position, neighbour) in trans_prob.keys():
                    probs.append(trans_prob[(position, neighbour)])
                else:
                    probs.append(0)
            if max(probs)==0:
                sampled_particle = [(x,y)]
            else:
                sampled_particle = random.choices(adj, weights=probs, k=1)
            # randomly sample from transitions based on transition probabilities
            # append only one particle, it cannot duplicate
            new_particles.append(sampled_particle[0])

        num = len(new_particles)

        weights = [1.]*num

        for i in range(num):
            (x,y) = new_particles[i]
            
            X = util.colToX(y)
            Y = util.rowToY(x)

            weights[i] = abs(util.pdf(sqrt((X-posX)**2 + (Y-posY)**2),std,observedDist))

        total_sum = 0.0
        for w in weights:
            total_sum += w

        particles = random.choices(new_particles, weights=weights, k=N)


        dictionary = {}

        for (x,y) in particles:
            if (x,y) in dictionary:
                dictionary[(x,y)] = dictionary[(x,y)] + 1
            else:
                dictionary[(x,y)] = 1
        
        for r in range(numRows):
            for c in range(numCols):
                if (r,c) in dictionary:
                    beliefGrid[r][c] = dictionary[(r,c)]/N
                else:
                    beliefGrid[r][c] = 0 
        self.belief.grid = beliefGrid
        self.belief.normalize()
        print(time.time() - begin)
        return
  
    def getBelief(self) -> Belief:
        return self.belief

   