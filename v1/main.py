"""In this function I define the GUI of the GridWorld environment."""
from environment import Environment
from policy_evaluation import iteration
from policy import policy

import numpy as np
import pygame


class GUI:
    """This is a class for the GUI of the GridWorld environment."""

    def __init__(self, size: int, terminalStates: list, screenHeight: int = 500):
        """This function initializes the GUI.

        Args:
            size (int): The size of the environment (size x size grid).
            terminalStates (list): The terminal states of the environment.
            screenHeight (int, optional): The height of the screen. Defaults to 500.
        """

        self.size = size
        self.env = Environment(size, terminalStates)
        self.screenHeight = screenHeight
        self.screenWidth = screenHeight * 2

        self.gridBorder = 2
        self.separation = 10
        self.screen = pygame.display.set_mode(
            (self.screenWidth + self.separation, self.screenHeight)
        )
        pygame.display.set_caption("GridWorld")

        # determine the font size
        fontSize = self.screenHeight // (self.size * 4)
        self.font = pygame.font.SysFont("Arial", fontSize)
        self.valueFunction = self.env.getInitialValueFunction()

    def drawGrid(self, topLeft, bottomRight, size, terminalColor=(0, 255, 0)):
        """This function draws a size by size grid on the screen.

        Args:
            topLeft (tuple): The coordinates of the top left corner of the grid.
            bottomRight (tuple): The coordinates of the bottom right corner of the grid.
            size (int): The size of the grid.
            terminalColor (tuple, optional): The color of the terminal states. Defaults to (255, 0, 0).
        """

        # calculate the width and height of each cell
        width = (bottomRight[0] - topLeft[0]) / size
        height = (bottomRight[1] - topLeft[1]) / size

        for i in range(size):
            for j in range(size):
                color = (0, 0, 0)
                border = self.gridBorder
                if self.env.coordinatesToIndex((i, j)) in self.env.terminalStates:
                    diff = np.array(
                        [
                            255 - terminalColor[0],
                            255 - terminalColor[1],
                            255 - terminalColor[2],
                        ],
                        dtype=np.float32,
                    )
                    diff *= 0.75
                    base = np.array(terminalColor, dtype=np.float32)
                    base += diff
                    background = tuple(base.astype(np.int32))

                    color = terminalColor
                    border *= 2
                    # draw the background of the terminal state
                    pygame.draw.rect(
                        self.screen,
                        background,
                        (
                            topLeft[0] + i * width + border,
                            topLeft[1] + j * height + border,
                            width - 2 * border,
                            height - 2 * border,
                        ),
                        0,
                    )

                    # calculate a lighter version of the color

                pygame.draw.rect(
                    self.screen,
                    color,
                    (topLeft[0] + i * width, topLeft[1] + j * height, width, height),
                    border,
                )

    def drawValueFunction(self):
        """Draws the current value function on the left half of the screen."""

        # draw the grid
        self.drawGrid((0, 0), (self.screenHeight, self.screenHeight), self.size)
        # draw the values
        for i in range(self.size):
            for j in range(self.size):
                val = round(self.valueFunction[i * self.size + j], 2)
                # draw the value in the middle of the cell
                text = self.font.render(str(val), True, (0, 0, 0))
                textRect = text.get_rect()
                self.screen.blit(
                    text,
                    (
                        i * self.screenHeight / self.size
                        + self.screenHeight / (2 * self.size)
                        - textRect.width / 2,
                        j * self.screenHeight / self.size
                        + self.screenHeight / (2 * self.size)
                        - textRect.height / 2,
                    ),
                )

    def getBestMoves(self, state):
        """Returns the indices of the best actions according to the current value function.

        Args:
            state (int): The state of the environment.

        Returns:
            list: The indices of the best actions.
        """
        if state in self.env.terminalStates:
            return []

        bestActions = []
        bestValue = -np.inf
        for action in range(len(self.env.actions)):
            nextState, reward, _ = self.env.makeMove(state, action)
            value = round(reward + self.valueFunction[nextState], 2)
            if value > bestValue:
                bestValue = value
                bestActions = [action]
            elif value == bestValue:
                bestActions.append(action)

        return bestActions

    def drawGreedyPolicy(self):
        """Draws the greedy policy on the right half of the screen."""
        self.drawGrid(
            (int(self.screenWidth / 2) + self.separation, 0),
            (self.screenWidth + self.separation, self.screenHeight),
            self.size,
        )

        # draw the arrows
        for i in range(self.size):
            for j in range(self.size):
                bestMoves = self.getBestMoves(i * self.size + j)
                for move in bestMoves:
                    # calculate the coordinates of the center of the cell
                    center = (
                        int(self.screenWidth / 2)
                        + self.separation
                        + i * self.screenHeight / self.size
                        + self.screenHeight / (2 * self.size),
                        j * self.screenHeight / self.size
                        + self.screenHeight / (2 * self.size),
                    )
                    # calculate the coordinates of the end of the arrow
                    if move == 0:
                        end = (
                            center[0],
                            center[1] - self.screenHeight / (2 * self.size),
                        )
                    elif move == 1:
                        end = (
                            center[0] + self.screenHeight / (2 * self.size),
                            center[1],
                        )
                    elif move == 2:
                        end = (
                            center[0],
                            center[1] + self.screenHeight / (2 * self.size),
                        )
                    elif move == 3:
                        end = (
                            center[0] - self.screenHeight / (2 * self.size),
                            center[1],
                        )

                    center, end = np.array(center), np.array(end)
                    newCenter = 0.75 * center + 0.25 * end
                    newEnd = 0.75 * end + 0.25 * center
                    center = newCenter
                    end = newEnd

                    # draw the arrow
                    pygame.draw.line(self.screen, (0, 0, 0), center, end, 3)

    def draw(self):
        """Draws the current frame on the screen."""

        # fill the screen with white
        self.screen.fill((255, 255, 255))

        # draw the current value function in the left half of the screen
        self.drawValueFunction()

        # draw the direction of a greedy policy in the right half of the screen
        self.drawGreedyPolicy()

    def setTerminals(self, pos):
        """Checks if the user clicked on a cell and if so, toggles it between terminal and non-terminal.

        Args:
            pos (tuple): The coordinates of the mouse.
        """

        pos = list(pos)
        # check if the user clicked on the left half of the screen
        if pos[0] < self.screenWidth / 2:
            pass
        elif pos[0] > self.screenHeight + self.separation:
            pos[0] -= self.screenHeight + self.separation
        else:
            return

        # calculate the coordinates of the cell
        cellX = int(pos[0] / (self.screenHeight / self.size))
        cellY = int(pos[1] / (self.screenHeight / self.size))
        cell = self.env.coordinatesToIndex((cellX, cellY))

        # if the cell is a terminal, remove it from the list of terminals
        if cell in self.env.terminalStates:
            self.env.terminalStates.remove(cell)
            self.valueFunction[cell] = np.random.rand()
        # if the cell is not a terminal, add it to the list of terminals
        else:
            self.env.terminalStates.add(cell)
            self.valueFunction[cell] = 0

    def mainLoop(self):
        """This function runs the main loop of the GUI."""

        running = True
        keyDown = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # if the user hits space do 10 iterations
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    for _ in range(10):
                        self.valueFunction, difference = iteration(
                            policy, self.valueFunction, self.env
                        )

                # if the user clicks on a cell toggle it between terminal and non-terminal
                if event.type == pygame.MOUSEBUTTONDOWN and keyDown == False:
                    keyDown = True
                    pos = pygame.mouse.get_pos()
                    self.setTerminals(pos)

                if event.type == pygame.MOUSEBUTTONUP:
                    keyDown = False

            self.draw()
            pygame.display.update()


if __name__ == "__main__":
    pygame.init()
    pygame.font.init()

    gui = GUI(
        10,
        [
            16,
            22,
            23,
            24,
            26,
            27,
            32,
            33,
            34,
            37,
            38,
            48,
            58,
            62,
            63,
            64,
            67,
            68,
            72,
            73,
            74,
            76,
            77,
            86,
        ],
        screenHeight=800,
    )
    gui.mainLoop()
