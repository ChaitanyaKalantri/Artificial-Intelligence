# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    """In dfs we have used a stack to provide us the required LIFO property since we need
    to traverse to deepest node possible."""
    fring = util.Stack()
    """We are maintaining the visited nodes using a dictionary."""
    visited = dict()
    """pushing the source node in the stack"""
    fring.push((problem.getStartState(), [],0,[]))
    """ Keep popping and processing the top of stack till elements in stack"""
    while True:
        if fring.isEmpty():
            return []

        state, path, cost, actions = fring.pop()
        """If goal state is reached return the path"""
        if problem.isGoalState(state):
            return actions
        """pushing the unvisited successors in the stack for processing first."""
        for successor, action, stepCost in problem.getSuccessors(state):
            visited[state]=True
            if successor not in visited:
                fring.push((successor, path + [successor], 0,actions + [action]))

    return actions

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    """In bfs we have used a queue to provide us the required FIFO property since we need
    to do a level order traversal."""
    fring = util.Queue()
    """We are maintaining the visited nodes using a dictionary."""
    visited = dict()

    fring.push((problem.getStartState(), [], 0, []))

    """ Keep getting and processing the queue till there are elements in the queue"""
    while True:
        if fring.isEmpty():
            return []

        state, path, cost, actions = fring.pop()

        visited[state] = True
       
	print state
	"""If goal state is reached return the path""" 
        if problem.isGoalState(state):
            return actions

        """pushing the unvisited successors in the queue for processing later."""
        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                visited[successor] = True
                fring.push((successor, path + [successor], 0, actions + [action]))

    return actions

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """In ucs we have used a priority queue to provide us the required priority retreival property since we need
    to traverse based on the cost function g."""
    fring = util.PriorityQueue()
    Expanded = []
    """We are maintaining the visited nodes using a dictionary."""
    visited = dict()
    fring.push((problem.getStartState(), []), 0)

    """ Keep getting and processing the queue till there are elements in the priority queue"""
    while True:
        if fring.isEmpty():
            return []

        state, movement = fring.pop()

        """If goal state is reached return the path""" 
        if problem.isGoalState(state):
            return movement

        """pushing the unvisited successors in the priority queue with updated cost for processing later."""
        if not state in Expanded:
            Expanded.append(state)
            for state, move, cost in problem.getSuccessors(state):
                fring.push((state, movement + [move]), problem.getCostOfActions(movement + [move]))

    return actions


    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    """In ucs we have used a priority queue to provide us the required priority retreival property since we need
    to traverse based on the cost function h."""
    fring = util.PriorityQueue()
    Expanded = []
    """We are maintaining the visited nodes using a dictionary."""
    visited = dict()
    h = heuristic(problem.getStartState(), problem)
    fring.push((problem.getStartState(), []), 0)

    """ Keep getting and processing the queue till there are elements in the priority queue"""
    while True:
        if fring.isEmpty():
            return []

        state, movement = fring.pop()

        """If goal state is reached return the path""" 
        if problem.isGoalState(state):
            return movement

        """pushing the unvisited successors in the priority queue with updated h function cost for processing later."""
        if not state in Expanded:
            Expanded.append(state)
            for state, move, cost in problem.getSuccessors(state):
                g_new = problem.getCostOfActions(movement + [move])
                h_new = heuristic(state, problem)
                fring.push((state, movement + [move]), g_new + h_new)

    return actions


    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
ucs = uniformCostSearch
astar = aStarSearch
