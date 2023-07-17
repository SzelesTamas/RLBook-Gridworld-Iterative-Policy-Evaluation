# Visualization for Gridworld policy evaluation example

## Version 1
This is an implementation of [this](http://incompleteideas.net/book/RLbook2020.pdf#page=99) Gridworld example from the book "Reinforcement Learning: An Introduction" by Sutton and Barto. The GUI is implemented using [PyGame](https://www.pygame.org/news). The backend is implemented using [NumPy](https://numpy.org/). In the example the Agent receives a reward of -1 for each move except in the terminal states where it receives 0. We want to evaluate an equiprobable random policy.