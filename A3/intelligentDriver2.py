'''
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
from math import sqrt
import util
import itertools
from turtle import Vec2D
from engine.const import Const
from engine.vector import Vec2d
from engine.model.car.car import Car
from engine.model.layout import Layout
from engine.model.car.junior import Junior
from configparser import InterpolationMissingOptionError

# Class: Graph
# -------------
# Utility class
class Graph(object):
    def __init__(self, nodes, edges, special_edges):
        self.nodes = nodes
        self.edges = edges
        self.special_edges = special_edges

# Class: IntelligentDriver
# ---------------------
# An intelligent driver that avoids collisions while visiting the given goal locations (or checkpoints) sequentially. 
class IntelligentDriver(Junior):

    # Funciton: Init
    def __init__(self, layout: Layout):
        self.visited_checkPoints = []
        self.burnInIterations = 30
        self.layout = layout 
        # self.worldGraph = None
        self.worldGraph = self.createWorldGraph()
        self.checkPoints = self.layout.getCheckPoints() # a list of single tile locations corresponding to each checkpoint
        self.transProb = util.loadTransProb()
        
    # ONE POSSIBLE WAY OF REPRESENTING THE GRID WORLD. FEEL FREE TO CREATE YOUR OWN REPRESENTATION.
    # Function: Create World Graph
    # ---------------------
    # Using self.layout of IntelligentDriver, create a graph representing the given layout.
    def createWorldGraph(self):
        nodes = []
        edges = []
        special_edges = []
        # create self.worldGraph using self.layout
        numRows, numCols = self.layout.getBeliefRows(), self.layout.getBeliefCols()

        # NODES #
        ## each tile represents a node
        nodes = [(x, y) for x, y in itertools.product(range(numRows), range(numCols))]
        
        # EDGES #
        ## We create an edge between adjacent nodes (nodes at a distance of 1 tile)
        ## avoid the tiles representing walls or blocks#
        ## YOU MAY WANT DIFFERENT NODE CONNECTIONS FOR YOUR OWN IMPLEMENTATION,
        ## FEEL FREE TO MODIFY THE EDGES ACCORDINGLY.

        ## Get the tiles corresponding to the blocks (or obstacles):
        blocks = self.layout.getBlockData()
        blockTiles = []
        #print(blocks)
        for block in blocks:
            row1, col1, row2, col2 = block[1], block[0], block[3], block[2] 
            # some padding to ensure the AutoCar doesn't crash into the blocks due to its size. (optional)
            #row1, col1, row2, col2 = row1-1, col1-1, row2+1, col2+1
            blockWidth = col2-col1 
            blockHeight = row2-row1 

            for i in range(blockHeight):
                for j in range(blockWidth):
                    blockTile = (row1+i, col1+j)
                    blockTiles.append(blockTile)

        ## Remove blockTiles from 'nodes'
        nodes = [x for x in nodes if x not in blockTiles]


        for node in nodes:
            x, y = node[0], node[1]
            adjNodes = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
            
            # only keep allowed (within boundary) adjacent nodes
            adjacentNodes = []
            nextToBlocked = []
            for tile in adjNodes:
                if tile[0]>=0 and tile[1]>=0 and tile[0]<numRows and tile[1]<numCols:
                    
                    if tile not in blockTiles:
                        if (tile[0]+1,tile[1]) in blockTiles or (tile[0]-1,tile[1]) in blockTiles or (tile[0],tile[1]+1) in blockTiles or (tile[0],tile[1]-1) in blockTiles:
                            nextToBlocked.append(tile)
                        else:
                            adjacentNodes.append(tile)


            for tile in adjacentNodes:
                edges.append((node, tile))
                edges.append((tile, node))
                            
            for tile in nextToBlocked:
                special_edges.append((node, tile))
                special_edges.append((tile,node))

        return Graph(nodes, edges, special_edges)

    #######################################################################################
    # Function: Get Next Goal Position
    # ---------------------
    # Given the current belief about where other cars are and a graph of how
    # one can driver around the world, chose the next position.
    #######################################################################################
    def getNextGoalPos(self, beliefOfOtherCars: list, parkedCars:list , chkPtsSoFar: int):
        '''
        Input:
        - beliefOfOtherCars: list of beliefs corresponding to all cars
        - parkedCars: list of booleans representing which cars are parked
        - chkPtsSoFar: the number of checkpoints that have been visited so far 
                       Note that chkPtsSoFar will only be updated when the checkpoints are updated in sequential order!
        
        Output:
        - goalPos: The position of the next tile on the path to the next goal location.
        - moveForward: Unset this to make the AutoCar stop and wait.

        Notes:
        - You can explore some files "layout.py", "model.py", "controller.py", etc.
         to find some methods that might help in your implementation. 
        '''
        def dist(pos1, pos2):
            return (pow(util.rowToY(pos1[0])-util.rowToY(pos2[0]), 2) + pow(util.colToX(pos1[1]) - util.colToX(pos2[1]), 2))*10000

        goalPos = None # next tile 
        for checkPoint in self.checkPoints:
                if checkPoint not in self.visited_checkPoints:
                    checkPointPos = checkPoint # next tile
                    break 
        moveForward = True
        #checkpoint pos is an integer tuple

        currPos = self.pos # the current 2D location of the AutoCar (refer util.py to convert it to tile (or grid cell) coordinate)
        #currPos is a float tuple, same as self.pos
        row = util.yToRow(currPos[1])
        col = util.xToCol(currPos[0])
        curr_node = (row, col)
        #curr_node is an integer tuple
        possible_positions = []
        if curr_node == checkPointPos:
            self.visited_checkPoints.append(checkPointPos)

        edges = self.worldGraph.edges
        special_edges = self.worldGraph.special_edges

        #an edge is an integer tuple
        for edge in edges:
            if edge[0]== curr_node:
                possible_positions.append(edge[1])
        
        # if len(possible_positions) == 0:
        #     for edge in special_edges:
        #         if edge[0] == curr_node:
        #             possible_positions.append(edge[1])

        vals = []
        vals2 = []
        threshold = 0.0000001                 #try changing this
        for i in range(len(possible_positions)):
            position = possible_positions[i]
            row, col = position
            prob = 0
            for belief_car in beliefOfOtherCars:
                prob = max(prob, belief_car.grid[row][col])
            vals.append([dist(position, checkPointPos), prob,i])
            vals2.append([prob ,dist(position, checkPointPos),i])        #to sort by prob
            if prob>threshold:
                moveForward = False                                 #if all positions are non-ideal, stop. ideal positions have probability of the order 1e-10 or below

        vals.sort()   
        vals2.sort()                                              #will sort according to the first index - distance from the next goal
        for val in vals:
            if val[1]<threshold:
                goalPos = possible_positions[val[2]]      #if a safe position is found
                break

        if goalPos==None:
            if len(vals2) != 0:                   
                goalPos = possible_positions[vals2[0][2]]     #go to the tile with the least probability of crash

        if goalPos ==None:
            moveForward = False
            goalPos = (0,0)
        else:    
            goalPos = (util.colToX(goalPos[1]), util.rowToY(goalPos[0]))
        
        return goalPos, moveForward

    # DO NOT MODIFY THIS METHOD !
    # Function: Get Autonomous Actions
    # --------------------------------
    def getAutonomousActions(self, beliefOfOtherCars: list, parkedCars: list, chkPtsSoFar: int):
        # Don't start until after your burn in iterations have expired
        if self.burnInIterations > 0:
            self.burnInIterations -= 1
            return[]
       
        goalPos, df = self.getNextGoalPos(beliefOfOtherCars, parkedCars, chkPtsSoFar)
        vectorToGoal = goalPos - self.pos
        wheelAngle = -vectorToGoal.get_angle_between(self.dir)
        driveForward = df
        actions = {
            Car.TURN_WHEEL: wheelAngle
        }
        if driveForward:
            actions[Car.DRIVE_FORWARD] = 1.0
        return actions
    
    