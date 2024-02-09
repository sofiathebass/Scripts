# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #ok my new strat 
        #calculate distance between me and the ghosts 
        #calculate distance between me and the food 
        #for i in successor game states calculate distance between me and the ghosts
        #for i in successor game states calculate distace between me and the food 
        stateGhost = 0
        stateFood = 0
        propGhost = 0
        propFood = 0
        closeFood = 999999
        closeGhost = 0
        closeGhostGhost = 0

        #calculate distance between me and the ghosts 
        lenGhosts = len(currentGameState.getGhostStates())
        if lenGhosts > 0:
            for i in range(1, lenGhosts):
                stateGhost += manhattanDistance(currentGameState.getGhostPosition(i), currentGameState.getPacmanPosition())
            stateGhost = stateGhost/lenGhosts

        #calculate distance between me and the food 
        if len(currentGameState.getFood().asList()) > 0:
            for i in currentGameState.getFood().asList():
                stateFood += manhattanDistance(i, currentGameState.getPacmanPosition())
            stateFood = stateFood/len(currentGameState.getFood().asList())
        
        #calculate total and closest distance between proposed successor and the ghosts, high distance is better so add
        lenGhosts = len(newGhostStates)
        if lenGhosts > 0:
            closeGhost = 999999
            for i in range(1, lenGhosts):
                dist = manhattanDistance(successorGameState.getGhostPosition(i), newPos)
                propGhost += dist 
                if dist < closeGhost: 
                    closeGhost = dist
                    closeGhostGhost = i
            propGhost = propGhost/lenGhosts
        
        #add second closest ghost to closeGhost to get closest two ghosts 
        if lenGhosts > 1:
            closeGhost2 = 999999
            for i in range(1, lenGhosts):
                dist = manhattanDistance(successorGameState.getGhostPosition(i), successorGameState.getGhostPosition(closeGhostGhost))
                if dist < closeGhost2: 
                    closeGhost2 = dist
            closeGhost += closeGhost2

        #calculate total and closest distance between proposed successor and all the food, low distance is better so subtract
        if len(newFood.asList()) > 0:
            for i in newFood.asList():
                dist = manhattanDistance(i, newPos)
                propFood += dist
                if dist < closeFood: 
                    closeFood = dist
            propFood = propFood/len(newFood.asList())
        else:
            closeFood = 0


        
        return 2*successorGameState.getScore() - closeFood + closeGhost

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        #pseudocode given by lecture 
        def value(state, depth, agent):
            if depth == 0 or state.isWin() or state.isLose():
                return [state.getScore(), ""]
            if agent == 0: 
                return maxValue(state, depth)
            else: 
                return minValue(state, agent, depth)
            
        def maxValue(state, depth):
            v = -999999999
            action = ""
            for i in state.getLegalActions(0): 
                betterState = [value(state.generateSuccessor(0, i), depth, 1)[0], i]
                if betterState[0] > v: 
                    v = betterState[0]
                    action = betterState[1]
            return [v, action]
        
        def minValue(state, agent, depth):
            v = 999999999
            action = ""
            next_agent = agent+1
            if state.getNumAgents() - 1 == agent:
                next_agent = 0
                depth = depth - 1
            for i in state.getLegalActions(agent): 
                betterState = [value(state.generateSuccessor(agent, i), depth, next_agent)[0], i]
                if betterState[0] < v: 
                    v = betterState[0]
                    action = betterState[1]
            return [v, action]
        
        return value(gameState, self.depth, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        #pseudocode given by lecture 
        def value(state, depth, agent, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                print("leaf:", state.getScore())
                return [state.getScore(), ""]
            if agent == 0: 
                return maxValue(state, depth, alpha, beta)
            else: 
                return minValue(state, agent, depth, alpha, beta)
            
        def maxValue(state, depth, alpha, beta):
            print("MAX alpha, beta", alpha, beta)
            v = -999999999
            action = ""
            for i in state.getLegalActions(0): 
                betterState = [value(state.generateSuccessor(0, i), depth, 1, alpha, beta)[0], i]
                if betterState[0] > v: 
                    v = betterState[0]
                    action = betterState[1]
                if v > beta: 
                    return [v, action]
                alpha = max(alpha, v)
            return [v, action]
        
        def minValue(state, agent, depth, alpha, beta):
            print("MIN alpha, beta", alpha, beta)
            v = 999999999
            action = ""
            next_agent = agent+1
            if state.getNumAgents() - 1 == agent:
                next_agent = 0
                depth = depth - 1
            for i in state.getLegalActions(agent): 
                betterState = [value(state.generateSuccessor(agent, i), depth, next_agent, alpha, beta)[0], i]
                if betterState[0] < v: 
                    v = betterState[0]
                    action = betterState[1]
                if v < alpha:
                    return [v, action]
                beta = min(beta, v)
            return [v, action]
        
        return value(gameState, self.depth, 0, -1000000, 1000000)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #pseudocode given by lecture 
        def value(state, depth, agent):
            if depth == 0 or state.isWin() or state.isLose():
                return [state.getScore(), ""]
            if agent == 0: 
                return maxValue(state, depth)
            else: 
                return expectValue(state, agent, depth)
            
        def maxValue(state, depth):
            v = -999999999
            action = ""
            for i in state.getLegalActions(0): 
                betterState = [value(state.generateSuccessor(0, i), depth, 1)[0], i]
                if betterState[0] > v: 
                    v = betterState[0]
                    action = betterState[1]
            return [v, action]
        
        def expectValue(state, agent, depth):
            v = 0
            p = 1/len(state.getLegalActions(agent))
            action = ""
            next_agent = agent+1
            if state.getNumAgents() - 1 == agent:
                next_agent = 0
                depth = depth - 1
            for i in state.getLegalActions(agent): 
                betterState = [value(state.generateSuccessor(agent, i), depth, next_agent)[0], i]
                v += p * betterState[0]
                action = betterState[1]
            return [v, action]
        
        return value(gameState, self.depth, 0)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    propGhost = 0
    propFood = 0
    closeFood = 999999
    closeGhost = 0
    closeGhostGhost = 0
    
    #calculate closest distance between state and the ghosts, high distance is better so add
    lenGhosts = len(currentGameState.getGhostStates())
    if lenGhosts > 0:
        closeGhost = 999999
        for i in range(1, lenGhosts):
            dist = manhattanDistance(currentGameState.getGhostPosition(i), currentGameState.getPacmanPosition())
            propGhost += dist 
            if dist < closeGhost: 
                closeGhost = dist
                closeGhostGhost = i
        propGhost = propGhost/lenGhosts
    
    #add second closest ghost to closeGhost to get closest two ghosts 
    if lenGhosts > 1:
        closeGhost2 = 999999
        for i in range(1, lenGhosts):
            dist = manhattanDistance(currentGameState.getGhostPosition(i), currentGameState.getGhostPosition(closeGhostGhost))
            if dist < closeGhost2: 
                closeGhost2 = dist
        closeGhost += closeGhost2

    #calculate total and closest distance between proposed successor and all the food, low distance is better so subtract
    if len(currentGameState.getFood.asList()) > 0:
        for i in currentGameState.getFood.asList():
            dist = manhattanDistance(i, currentGameState.getPacmanPosition())
            propFood += dist
            if dist < closeFood: 
                closeFood = dist
        propFood = propFood/len(currentGameState.getFood.asList())
    else:
        closeFood = 0

    print("each move", currentGameState.getScore(), closeFood, closeGhost)

    return currentGameState.getScore() - closeFood + closeGhost

# Abbreviation
better = betterEvaluationFunction
