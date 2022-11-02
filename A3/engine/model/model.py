from .car.car import Car
from .car.agent import Agent
from engine.const import Const
from .car.junior import Junior
from engine.vector import Vec2d
from autoDriver import AutoDriver
from engine.model.block import Block
from intelligentDriver import IntelligentDriver
from engine.model.agentCommunication import AgentCommunication


import copy
import util
import random
import threading

class Model(object):

    def __init__(self, layout):
        self._initBlocks(layout)
        self._initIntersections(layout)
        self.layout = layout
        startX = layout.getStartX()
        startY = layout.getStartY()
        startDirName = layout.getJuniorDir()
        # self.junior = AutoDriver()
        if Const.INTELLIGENT_DRIVER:
            self.junior = IntelligentDriver(layout)
        else:
            self.junior = AutoDriver()
            
        self.junior.setup(
            Vec2d(startX, startY), 
            startDirName, 
            Vec2d(0, 0)
        )
        self.cars = [self.junior]
        self.otherCars = [] 

        self.visited = [0]*Const.NUM_CHECKPTS # Const.NUM_CHECKPTS == len(self.finish)
        self.nextCheckPtIdx = 0

        if not Const.MULTIPLE_GOALS:
            self.finish = Block(layout.getFinish()) 
        else:
            self.finish = []
            for block in layout.getFinish():
                # print(block)
                self.finish.append(Block(block))

        agentComm = AgentCommunication()
        agentGraph = layout.getAgentGraph()
        for _ in range(Const.NUM_AGENTS):
            startNode = self._getStartNode(agentGraph)
            # other = Agent(startNode, layout.getAgentGraph(), self, agentComm) 
            other = Agent(startNode, layout.getAgentGraph(), self, agentComm, Const.CARS_PARKED)
            self.cars.append(other)
            self.otherCars.append(other)
        self.observations = []
        agentComm.addAgents(self.otherCars)
        self.modelLock = threading.Lock()
        self.probCarSet = False
        
        
    def _initBlocks(self, layout):
        self.blocks = []
        for blockData in layout.getBlockData():
            block = Block(blockData)
            self.blocks.append(block)
            
    def _initIntersections(self, layout):
        self.intersections = []
        for blockData in layout.getIntersectionNodes():
            block = Block(blockData)
            self.intersections.append(block)
            
    def _getStartNode(self, agentGraph):
        while True:
            node = agentGraph.getRandomNode()
            pos = node.getPos()
            alreadyChosen = False
            for car in self.otherCars:
                if car.getPos() == pos:
                    alreadyChosen = True
                    break
            if not alreadyChosen: 
                return node
            
    def checkVictory(self):
        bounds = self.junior.getBounds()
        for point in bounds:
            if self.finish.containsPoint(point.x, point.y): return True
        return False

    def unordered_checkVictory(self): 
        bounds = self.junior.getBounds() 

        for idx, checkpt in enumerate(self.finish):
            for point in bounds:
                if checkpt.containsPoint(point.x, point.y):
                    if self.visited[idx]==0:
                        print(f"Checkpoint {idx} visited!")
                    self.visited[idx] = 1

        # print(visited)
        if self.visited==[1]*len(self.finish):
            return True 
        return False
    
    # for ordered visit of checkpoints
    def _checkVictory(self):
        bounds = self.junior.getBounds() 
        checkpt = self.finish[self.nextCheckPtIdx]

        for point in bounds:
            if checkpt.containsPoint(point.x, point.y):
                if self.visited[self.nextCheckPtIdx]==0:
                    print(f"Checkpoint {self.nextCheckPtIdx+1} visited!")
                self.visited[self.nextCheckPtIdx] = 1
                self.nextCheckPtIdx += 1
                break
        
        if self.visited==[1]*Const.NUM_CHECKPTS:
            Const.COMPLETED_CHECKPTS += 1 # the final checkpoint is not counted by 'getNextGoalPos'
            return True 
        return False

    def checkCollision(self, car):
        bounds = car.getBounds()
        # check for collision with fixed obstacles
        for point in bounds:
            if not self.inBounds(point.x, point.y): return True
        
        # check for collision with other cars
        for other in self.cars:
            if other == car: continue
            if other.collides(car.getPos(), bounds): return True
        return False
        
    def getIntersection(self, x, y):
        for intersection in self.intersections:
            if intersection.containsPoint(x, y): return intersection
        return None
        
    def inIntersection(self, x, y):
        return self.getIntersection(x, y) != None
            
    def inBounds(self, x, y):
        if x < 0 or x >= self.getWidth(): return False
        if y < 0 or y >= self.getHeight(): return False
        for block in self.blocks:
            if block.containsPoint(x, y): return False
        return True
    
    def getWidth(self):
        return self.layout.getWidth()
    
    def getHeight(self):
        return self.layout.getHeight()
    
    def getBeliefRows(self):
        return self.layout.getBeliefRows()
    
    def getBeliefCols(self):
        return self.layout.getBeliefCols()
            
    def getBlocks(self):
        return self.blocks
    
    def getFinish(self):
        return self.finish
        
    def getCars(self):
        return self.cars
    
    def getOtherCars(self):
        return self.otherCars
    
    def getJunior(self):
        return self.junior
    
    def getAgentGraph(self):
        return self.layout.getAgentGraph()
    
    def getJuniorGraph(self):
        return self.layout.getJuniorGraph()
    
    # to calculate the probability of presence of any StdCar on the grid cells
    def setProbCar(self, beliefs):
        self.currBeliefs = beliefs

        self.modelLock.acquire()
        total = util.Belief(self.getBeliefRows(), self.getBeliefCols(), 0.0)
        for r in range(self.getBeliefRows()):
            for c in range(self.getBeliefCols()):
                pNot = 1.0
                for b in beliefs:
                    carP = b.getProb(r, c)
                    pNot *= (1.0 - carP)
                p = 1.0 - pNot
                total.setProb(r, c, p)
        self.probCar = total
        self.modelLock.release()
        self.probCarSet = True
    
    def _getProbCar(self):
        if not self.probCarSet: return None
        self.modelLock.acquire()
        probCar = copy.deepcopy(self.probCar)
        self.modelLock.release()
        return probCar
    
    def getProbCar(self):
        if not getattr(self,"currBeliefs", False):
            return None 
        self.modelLock.acquire()
        probCar = copy.deepcopy(self.currBeliefs)
        self.modelLock.release()
        return probCar

