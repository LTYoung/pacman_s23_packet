from pacai.agents.capture.capture import CaptureAgent
from pacai.core.eval import score as evalFn
from pacai.util import util
from pacai.core.directions import Directions
import logging
import random
import time

"""shared agent infrastructure
"""


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions.
    """

    def __init__(self, index, evalFn=evalFn, depth=2, **kwargs):
        super().__init__(index, **kwargs)
        self._evaluationFunction = evalFn
        self._treeDepth = int(depth)

    def getEvaluationFunction(self):
        return self._evaluationFunction

    def getTreeDepth(self):
        return self._treeDepth

    def maxValue(self, state, depth, angentIndex, alpha, beta):
        score = -float("inf")
        scores = []
        legalActions = state.getLegalActions(self.index)
        if "Stop" in legalActions:
            legalActions.remove("Stop")

        if depth == 0 or state.isWin() or state.isLose():
            score = self.getEvaluationFunction()(state)  # type: ignore
            # return a random action
            index = random.randint(0, len(legalActions) - 1)
            return score, legalActions[index]

        scores = [
            self.minValue(
                state.generateSuccessor(self.index, action),
                depth,
                angentIndex,
                alpha,
                beta,
            )[0]
            for action in legalActions
        ]
        score = max(scores)
        indexies = []
        for i in range(len(scores)):
            if scores[i] == score:
                indexies.append(i)
        index = indexies[random.randint(0, len(indexies) - 1)]

        alpha = max(alpha, score)

        return score, legalActions[index]

    def minValue(self, state, depth, angentIndex, alpha, beta):
        score = float("inf")
        scores = []
        legalActions = state.getLegalActions(self.index)
        if "Stop" in legalActions:
            legalActions.remove("Stop")

        if depth == 0 or state.isWin() or state.isLose():
            score = self.getEvaluationFunction()(state)  # type: ignore
            # return a random action
            index = random.randint(0, len(legalActions) - 1)
            return score, legalActions[index]

        if angentIndex != state.getNumAgents() - 1:
            scores = [
                self.minValue(
                    state.generateSuccessor(self.index, action),
                    depth,
                    angentIndex + 1,
                    alpha,
                    beta,
                )[0]
                for action in legalActions
            ]
        else:
            scores = [
                self.maxValue(
                    state.generateSuccessor(self.index, action),
                    depth - 1,
                    0,
                    alpha,
                    beta,
                )[0]
                for action in legalActions
            ]
        score = min(scores)
        indexies = []
        for i in range(len(scores)):
            if scores[i] == score:
                indexies.append(i)
        index = indexies[random.randint(0, len(indexies) - 1)]

        beta = min(beta, score)
        return score, legalActions[index]

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """
        depth = self.getTreeDepth()

        # find out what team we are on
        team = self.getTeam(gameState)

        action = self.maxValue(gameState, depth, team[0], -float("inf"), float("inf"))
        return action[1]

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if pos != util.nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        stateEval = sum(features[feature] * weights[feature] for feature in features)

        return stateEval

    def getFeatures(self, gameState, action):
        """
        Returns a dict of features for the state.
        The keys match up with the return from `ReflexCaptureAgent.getWeights`.
        """

        successor = self.getSuccessor(gameState, action)

        return {"successorScore": self.getScore(successor)}

    def getWeights(self, gameState, action):
        """
        Returns a dict of weights for the state.
        The keys match up with the return from `ReflexCaptureAgent.getFeatures`.
        """

        return {"successorScore": 1.0}


class CombinedReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features["successorScore"] = self.getScore(successor)

        foodList = self.getFood(successor).asList()

        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features["distanceToFood"] = minDistance

        features["onDefense"] = 1
        if myState.isPacman():
            features["onDefense"] = 0

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features["numInvaders"] = len(invaders)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features["invaderDistance"] = min(dists)

        if action == Directions.STOP:
            features["stop"] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if action == rev:
            features["reverse"] = 1

        return features

    def getWeights(self, gameState, action):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        if len(invaders) > 0:
            return {
                "successorScore": 1,
                "distanceToFood": 0,
                "numInvaders": -1000,
                "onDefense": 3,
                "invaderDistance": -10,
                "stop": -1000,
                "reverse": -2,
            }
        else:
            return {
                "successorScore": 100,
                "distanceToFood": -1,
                "numInvaders": -1000,
                "onDefense": 0,
                "invaderDistance": -10,
                "stop": -1000,
                "reverse": -2,
            }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features["onDefense"] = 1
        if myState.isPacman():
            features["onDefense"] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features["numInvaders"] = len(invaders)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features["invaderDistance"] = min(dists)

        if action == Directions.STOP:
            features["stop"] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if action == rev:
            features["reverse"] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            "numInvaders": -1000,
            "onDefense": 100,
            "invaderDistance": -10,
            "stop": -100,
            "reverse": -2,
        }
