import util 
from util import Belief, pdf 
from engine.const import Const
from math import sin,cos,sqrt
import random

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

        numRows = self.belief.numRows
        numCols = self.belief.numCols

        # print(self.transProb)        
        std = Const.SONAR_STD

        trans_prob = self.transProb
        
        #setting the belief according to particle filter


        N = numRows*numCols*2
        
        beliefGrid = self.belief.grid

        particles = []

        for row in range(numRows):
            for col in range(numCols):
                number = int(beliefGrid[row][col]*N)
                print(beliefGrid[row][col])
                for _ in range(number):
                    particles.append((row,col))

        iter = 0

        max_iter = 10

        print(particles)

        while(iter < max_iter):

            new_particles = []

            for (x,y) in particles:
                adj = [(x,y+1),(x,y-1),(x+1,y),(x-1,y),(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)]
                probs = [(trans_prob[(x,y),adj[c]] if ((x,y),adj[c]) in trans_prob else 0,adj[c]) for c in range(len(adj))]
                probs.sort()
                #appending only 1 new particle based on direction of max probability

                new_particles.append(probs[-1][1])
                new_particles.append(probs[-2][1])
                new_particles.append(probs[-3][1])
            
            print(new_particles)

            num = len(new_particles)

            weights = [1.]*num

            for i in range(num):
                (x,y) = new_particles[i]
                
                X = util.colToX(x)
                Y = util.rowToY(y)

                weights[i] = abs(util.pdf(sqrt((X-posX)**2 + (Y-posY)**2),std,observedDist))

            total_sum = 0.0
            for w in weights:
                total_sum += w
            
            print(weights)

            for i in range(num):
                weights[i]/=total_sum

            particles = random.choices(new_particles, weights=weights, k=N)

            iter += 1

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
                    beliefGrid[r][c] = 1e-3   
        self.belief.grid = beliefGrid
        self.belief.normalize()
        return
  
    def getBelief(self) -> Belief:
        return self.belief

   