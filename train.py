from agent import Agent
from maze import Maze

a = [0, 0, 0, 0, 0, 0,
     0, 0, 1, 1, 1, 0,
     0, 0, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0,
     0, 0, 0, 0, 1, 0,
     0, 0, 0, 0, 1, 2]

maze = Maze()
maze.set_wall(a)
print(maze.grid)
agent = Agent(maze)
agent.pretraining_heatmap()
agent.learn()

