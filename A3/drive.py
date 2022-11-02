'''
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
from engine.const import Const
from engine.controller import Controller
from engine.view.display import Display

import sys
import optparse
import random
import signal
import time
import datetime
import os

if __name__ == '__main__':
    
    parser = optparse.OptionParser()
    parser.add_option('-p', '--parked', dest='parked', default=False, action='store_true')
    parser.add_option('-d', '--display', dest='display', default=False, action='store_true')
    parser.add_option('-k', '--numCars', type='int', dest='numCars', default=3)
    parser.add_option('-l', '--layout', dest='layout', default='small')
    parser.add_option('-i', '--inference', dest='inference', default='exactInference')
    parser.add_option('-s', '--speed', dest='speed', default='verySlow')
    parser.add_option('-a', '--auto', dest='auto', default=False, action='store_true')
    parser.add_option('-f', '--fixedSeed', dest='fixedSeed', default=False, action='store_true')
    parser.add_option('-m', '--checkpoints', dest='checkpoints', default=False, action='store_true')
    parser.add_option('-j', '--intelligentDriver', dest='intelligentDriver', default=False, action='store_true')

    (options, _) = parser.parse_args()
    
    Const.WORLD = options.layout
    Const.CARS_PARKED = options.parked
    Const.SHOW_CARS = options.display
    Const.NUM_AGENTS = options.numCars
    Const.INFERENCE = options.inference
    Const.SPEED = options.speed
    Const.HEARTBEATS_PER_SECOND = Const.HEARTBEAT_DICT[Const.SIM_SPEED]
    Const.SECONDS_PER_HEARTBEAT = 1.0 / Const.HEARTBEATS_PER_SECOND
    Const.AUTO = options.auto

    Const.INTELLIGENT_DRIVER = options.intelligentDriver
    Const.MULTIPLE_GOALS = options.checkpoints
    if options.checkpoints:
        Const.WORLD = 'm_'+str(Const.WORLD)
        if Const.WORLD=='m_small':
            Const.NUM_CHECKPTS = 2
        elif Const.WORLD=='m_lombard':
            Const.NUM_CHECKPTS = 3
        elif Const.WORLD=='m_large':
            Const.NUM_CHECKPTS = 4
    
    if Const.INTELLIGENT_DRIVER and not Const.MULTIPLE_GOALS:
        print("Please switch to multiple goals layout for testing Intelligent Driver!")
        print('closing...')
        Display.endGraphics()

    
    # Fix the random seed
    if options.fixedSeed: random.seed('driverlessCar')

    controller = Controller()
    start = time.time()
    quit = controller.drive()
    end = time.time()
    if not quit:
        controller.freezeFrame()
        print(f"Simulation time: {end-start} seconds")

    print('closing...')
    Display.endGraphics()
