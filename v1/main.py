"""In this function I define the GUI of the GridWorld environment."""
from environment import Environment
from policy_evaluation import iteration
from policy import policy

import numpy as np
import pygame

class GUI:
    """This is a class for the GUI of the GridWorld environment."""
    
    def __init__(self, size:int, terminalStates:list, screenHeight: int = 500):
        """This function initializes the GUI.
        
        Args:
            size (int): The size of the environment (size x size grid).
            terminalStates (list): The terminal states of the environment.
            screenHeight (int, optional): The height of the screen. Defaults to 500.
        """
        
        self.size = size
        self.terminals = terminalStates
        self.env = Environment(size, terminalStates)
        self.screenHeight = screenHeight
        self.screenWidth = screenHeight*2
        
        self.screen = pygame.display.set_mode((self.screenHeight, self.screenWidth))
        self.font = pygame.font.SysFont('Arial', 20)
        self.valueFunction = self.env.getInitialValueFunction()
        self.gridBorder = 10
        
    def drawGrid(self, topLeft, bottomRight, size):
        """This function draws a size by size grid on the screen.
        
        Args:
            topLeft (tuple): The coordinates of the top left corner of the grid.
            bottomRight (tuple): The coordinates of the bottom right corner of the grid.
            size (int): The size of the grid.
        """
        
        # calculate the width and height of each cell
        width = (bottomRight[0] - topLeft[0]) / size
        height = (bottomRight[1] - topLeft[1]) / size
        
        for i in range(size):
            for j in range(size):
                pygame.draw.rect(self.screen, (0, 0, 0), (topLeft[0] + i*width, topLeft[1] + j*height, width, height), self.gridBorder)
        
        
        
    def drawValueFunction(self):
        """Draws the current value function on the left half of the screen."""
        self.drawGrid((0, 0), (self.screenHeight, self.screenHeight), self.size)
        
        
    def drawGreedyPolicy(self):
        """Draws the greedy policy on the right half of the screen."""
        self.drawGrid((int(self.screenWidth/2), 0), (self.screenWidth, self.screenHeight), self.size)
        
    def draw(self):
        """Draws the current frame on the screen."""
        
        # fill the screen with white
        self.screen.fill((255, 255, 255))
        
        # draw the current value function in the left half of the screen
        self.drawValueFunction()
        
        # draw the direction of a greedy policy in the right half of the screen
        self.drawGreedyPolicy()
        
        
    def mainLoop(self):
        """This function runs the main loop of the GUI."""
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # if the user hits space do an iteration
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.env = self.iteration(policy, self.env, self.env)
            
            self.draw()
            pygame.display.update()
        
        
if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    
    gui = GUI(4, [0, 15])
    gui.mainLoop()