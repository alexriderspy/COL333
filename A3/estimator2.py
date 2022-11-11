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

        # numRows = self.belief.numRows
        # numCols = self.belief.numCols

        # # print(self.transProb)        
        # std = Const.SONAR_STD

        # val = 0.0
        # delta = 0.0006
        # for _ in range(10000):

        #     X = observedDist*cos(val) + posX
        #     Y = observedDist*sin(val) + posY

        #     val += delta

        #     rowSouth = util.yToRow(Y+std)
        #     rowNorth = util.yToRow(Y-std)

        #     colEast = util.xToCol(X+std)
        #     colWest = util.xToCol(X-std)

        #     for row in range(max(0,rowNorth), min(numRows,rowSouth+1)):
        #         for col in range(max(0,colWest), min(numCols,colEast+1)):
        #             self.belief.addProb(row,col,100000)
        
        # self.belief.normalize()

        numRows = self.belief.numRows
        numCols = self.belief.numCols

        # print(self.transProb)        
        std = Const.SONAR_STD

        trans_prob = self.transProb
        
        #setting the belief according to exact inference

        alpha = 70
        #beta = 4
        for _ in range(200):
            for row in range(numRows):
                for col in range(numCols):
                    prob = 100.0
                    if row+1 < numRows:
                        if ((row+1,col),(row,col)) in trans_prob:
                            prob += trans_prob[(row+1,col),(row,col)]*self.belief.getProb(row+1,col)*alpha
                    if col+1 < numCols:
                        if ((row,col+1),(row,col)) in trans_prob:
                            prob += trans_prob[(row,col+1),(row,col)]*self.belief.getProb(row,col+1)*alpha
                    if row-1 >= 0:
                        if ((row-1,col),(row,col)) in trans_prob:
                            prob += trans_prob[(row-1,col),(row,col)]*self.belief.getProb(row-1,col)*alpha
                    if col-1 >= 0:
                        if ((row,col-1),(row,col)) in trans_prob:
                            prob += trans_prob[(row,col-1),(row,col)]*self.belief.getProb(row,col-1)*alpha
                    
                    if row+1 < numRows and col+1 < numCols:
                        if ((row+1,col+1),(row,col)) in trans_prob:
                            prob += trans_prob[((row+1,col+1),(row,col))]*self.belief.getProb(row+1,col+1)*alpha
                    if row+1 < numRows and col-1 >=0:
                        if ((row+1,col-1),(row,col)) in trans_prob:
                            prob += trans_prob[((row+1,col-1),(row,col))]*self.belief.getProb(row+1,col-1)*alpha
                    if row-1 >=0 and col+1 < numCols:
                        if ((row-1,col+1),(row,col)) in trans_prob:
                            prob += trans_prob[((row-1,col+1),(row,col))]*self.belief.getProb(row-1,col+1)*alpha
                    if row-1 >= 0 and col-1 >= 0:
                        if ((row-1,col-1),(row,col)) in trans_prob:
                            prob += trans_prob[((row-1,col-1),(row,col))]*self.belief.getProb(row-1,col-1)*alpha
                    

                    if ((row,col),(row,col)) in trans_prob:
                        prob += trans_prob[(row,col),(row,col)] * self.belief.getProb(row,col)*alpha
                    
                    prob = prob * (util.pdf(sqrt((util.colToX(col)-posX)**2 + (util.rowToY(row)-posY)**2),std,observedDist))
                    #print(prob)
                    self.belief.setProb(row,col,prob)

        self.belief.normalize()
        return
  
    def getBelief(self) -> Belief:
        return self.belief

   