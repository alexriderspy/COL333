'''
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
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
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

# Class: IntelligentDriver
# ---------------------
# An intelligent driver that avoids collisions while visiting the given goal locations (or checkpoints) sequentially. 
class IntelligentDriver(Junior):

    # Funciton: Init
    def __init__(self, layout: Layout):
        self.burnInIterations = 30
        self.layout = layout 
        # self.worldGraph = None
        self.worldGraph = self.createWorldGraph()
        self.checkPoints = self.layout.getCheckPoints() # a list of single tile locations corresponding to each checkpoint
        self.transProb = util.loadTransProb()
        self.blockTiles = []
        self.visitedCheckPoints= []
        self.lastpath = []
        
    # ONE POSSIBLE WAY OF REPRESENTING THE GRID WORLD. FEEL FREE TO CREATE YOUR OWN REPRESENTATION.
    # Function: Create World Graph
    # ---------------------
    # Using self.layout of IntelligentDriver, create a graph representing the given layout.
    def createWorldGraph(self):
        nodes = []
        edges = {}
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
        for block in blocks:
            row1, col1, row2, col2 = block[1], block[0], block[3], block[2] 
            # some padding to ensure the AutoCar doesn't crash into the blocks due to its size. (optional)
            row1, col1, row2, col2 = row1-1, col1-1, row2+1, col2+1
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
            for tile in adjNodes:
                if tile[0]>=0 and tile[1]>=0 and tile[0]<numRows and tile[1]<numCols:
                    if tile not in blockTiles:
                        adjacentNodes.append(tile)

            for tile in adjacentNodes:
                if node not in edges:
                    edges[node] = [tile]
                else:
                    edges[node].append(tile)
                if tile not in edges:
                    edges[tile] = [node]
                else:
                    edges[tile].append(node)
                #THIS LINE WAS IMPORTANT, BOTH SIDE EDGES NEEDED
        return Graph(nodes, edges)

    #######################################################################################
    # Function: Get Next Goal Position
    # ---------------------
    # Given the current belief about where other cars are and a graph of how
    # one can driver around the world, chose the next position.
    #######################################################################################

    def bfs(self, start, end):
        m = self.layout.getBeliefRows()
        n = self.layout.getBeliefCols()
        parent = [[None for _ in range(n)] for _ in range(m)]
        edges = self.worldGraph.edges
        visited = [[False for _ in range(n)] for _ in range(m)]
        visited[start[0]][start[1]] = True
        q = [start]
        while len(q):
            node = q.pop(0)
            if node in edges:
                for next_node in edges[node]:
                    if visited[next_node[0]][next_node[1]] == False:
                        q.append(next_node)
                        visited[next_node[0]][next_node[1]] = True
                        parent[next_node[0]][next_node[1]] = node
        path = []
        cur_node = end
        while cur_node:
            path.append(cur_node)
            cur_node = parent[cur_node[0]][cur_node[1]]
        path.reverse()
        return path

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
            return (pow(pos1[0]-pos2[0], 2) + pow(pos1[1] - pos2[1], 2))

        self.worldGraph = self.createWorldGraph()
        goalPos = (0,0)
        moveForward = True

        currPos = self.getPos() # the current 2D location of the AutoCar (refer util.py to convert it to tile (or grid cell) coordinate)
        # BEGIN_YOUR_CODE 

        row = util.yToRow(currPos[1])
        col = util.xToCol(currPos[0])

        curr_node = (row,col)
        goalPos = curr_node

        for checkPoint in self.checkPoints:
            if checkPoint not in self.visitedCheckPoints:
                checkPointPos = checkPoint # next tile
                break 

        if curr_node == checkPointPos:
            self.visitedCheckPoints.append(curr_node)


        goalPos = checkPointPos
        path_to_goal = self.bfs(curr_node, goalPos)

        if len(path_to_goal)>1:
            goalPos = path_to_goal[1]
            """self.lastpath = path_to_goal
        else:
            #the case where the autocar deviates from its path because of the stdcar
            #find the nearest node in the last path calculated
            path = self.lastpath
            min_dist = 10000000
            min_node = None
            for node in path:
                d = dist(node, curr_node)
                if d<min_dist:
                    min_dist = d
                    min_node = node
            if len(path)==0:
                goalPos = checkPointPos
            else:
                goalPos = min_node"""
        def get_probability(belief, row, col):
            #currently only considering 4 directions of movement, up down left and right
            numRows, numCols = self.layout.getBeliefRows(), self.layout.getBeliefCols()
            row1 = max(row-1, 0)
            row2 = min(row+1, numRows-1)
            col1 = max(col-1, 0)
            col2 = min(col+1, numCols-1)
            combinations = []
            combinations.append((row1, col))
            combinations.append((row2, col))
            combinations.append((row, col1))
            combinations.append((row, col2))

            prob = 0
            for i in range(4):
                row_old, col_old = combinations[i]
                transition = ((row_old, col_old), (row, col))
                if transition in self.transProb.keys():
                    prob+= self.transProb[transition]*belief[row_old][col_old]
            return prob

        def dist(pos1, pos2):
            return (pow(pos1[0]-pos2[0], 2) + pow(pos1[1] - pos2[1], 2))

        for checkPoint in self.checkPoints:
            if checkPoint not in self.visitedCheckPoints:
                checkPointPos = checkPoint # next tile
                break 

        if curr_node == checkPointPos:
            self.visitedCheckPoints.append(curr_node)
        
        if row == checkPointPos[0]:
            goalPos = (checkPointPos[0],col+1)


        edges = self.worldGraph.edges
        possible_positions = []

        for edge in edges:
            if edge[0]== curr_node:
                possible_positions.append(edge[1])

        vals = []
        vals2 = []
        threshold = 0.0000001
        go_to_safe = False
        for i in range(len(possible_positions)):
            position = possible_positions[i]
            row, col = position
            prob = 0
            for belief_car in beliefOfOtherCars:
                prob = max(prob, get_probability(belief_car.grid, row, col))
            
            distance = dist(position, checkPointPos)
            vals.append([distance, prob,i])
            vals2.append([prob , distance,i])        #to sort by prob
            if prob>threshold:
                go_to_safe = True                                  #if all positions are non-ideal, stop. ideal positions have probability of the order 1e-10 or below

        vals2.sort()
        if go_to_safe:
            if len(vals2) != 0:                   
                goalPos = possible_positions[vals2[0][2]]     #go to the tile with the least probability of crash
        
        goalPos = (util.colToX(goalPos[1]), util.rowToY(goalPos[0]))
        # END_YOUR_CODE
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
    
    