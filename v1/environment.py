"""In this file I define the GridWorld environment."""
import numpy as np

class Environment:
    """This class represents the GridWorld environment."""
    
    def __init__(self, size: int, terminalStates: list):
        """This function initializes the environment.
        
        Args:
            size (int): The size of the environment (size x size grid).
            terminalStates (list): The terminal states of the environment.
        """
        self.size = size
        self.stateCount = size * size
        self.terminalStates = {*terminalStates}
        # The actions are represented as follows:
        # 0: up
        # 1: right
        # 2: down
        # 3: left
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
    def indexToCoordinates(self, index: int) -> tuple:
        """This function converts an index to coordinates.
        
        Args:
            index (int): The index to convert.
            
        Returns:
            tuple: The coordinates of the index.
        """
        return (index // self.size, index % self.size)
    
    def coordinatesToIndex(self, coordinates: tuple) -> int:
        """This function converts coordinates to an index.
        
        Args:
            coordinates (tuple): The coordinates to convert.
            
        Returns:
            int: The index of the coordinates.
        """
        return coordinates[0] * self.size + coordinates[1]
    
    def isInside(self, coordinates: tuple) -> bool:
        """Returns whether the coordinates are inside the environment.
        
        Args:
            coordinates (tuple): The coordinates to check.
        
        Returns:
            bool: Whether the coordinates are inside the environment.
        """
        return 0 <= coordinates[0] < self.size and 0 <= coordinates[1] < self.size
        
    def makeMove(self, startState:int, action:int):
        """This function makes a move in the environment.
        
        Args:
            startState (int): The state to start from.
            action (int): The action to take.
            
        Returns:
            tuple: (endState, reward, isTerminal)
        """
        
        startCoordinates = self.indexToCoordinates(startState)
        endCoordinates = (startCoordinates[0] + self.actions[action][0], startCoordinates[1] + self.actions[action][1])
        if(self.isInside(endCoordinates)):
            endState = self.coordinatesToIndex(endCoordinates)
            if(endState in self.terminalStates):
                return (endState, -1, True)
            return (endState, -1, False)
        else:
            return (startState, -1, False)
        
    def getInitialValueFunction(self) -> np.array:
        """This function returns the initial value function. All random except the terminal states which are 0.
        
        Returns:
            np.array: The initial value function.
        """
        valueFunction = np.random.rand(self.stateCount)
        for terminalState in self.terminalStates:
            valueFunction[terminalState] = 0
        return valueFunction