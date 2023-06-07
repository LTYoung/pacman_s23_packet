from pacai.agents.capture.capture import CaptureAgent
from pacai.core.eval import score as evalFn
from pacai.util import util
from pacai.core.directions import Directions
import random

"""shared agent infrastructure
"""


class GigaChadAgent(CaptureAgent):
    """
    Agent infra like searching on tree and actions to take for
    both offensive and defensive agents
    """

    # using default evaluation function
    # "This evaluation function is meant for use with adversarial search agents
    # (not reflex agents)."
    def __init__(self, index, evalFn=evalFn, depth=2, **kwargs):
        super().__init__(index, **kwargs)

        self._evalFn = evalFn
        self._depth = depth
        self.agent = CaptureAgent  # just gonna use this for now

    def getEvalFn(self):
        return self._evalFn

    def getTreeDepth(self):
        return self._depth

    # From Young's AB implementation
    def getAction(self, gameState):
        depth = self.getTreeDepth()

        return self.maxValue(gameState, depth, 0, -float("inf"), float("inf"))[1]

    def maxValue(self, state, depth, angentIndex, alpha, beta):
        score = -float("inf")
        scores = []
        legalActions = state.getLegalActions()

        if depth == 0 or state.isWin() or state.isLose():
            score = self.getEvaluationFunction()(state)  # type: ignore
            return score, "Stop"

        if "Stop" in legalActions:
            legalActions.remove("Stop")

        scores = [
            self.minValue(state.generateSuccessor(0, action), depth, 1, alpha, beta)[0]
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
        legalActions = state.getLegalActions(angentIndex)

        if depth == 0 or state.isWin() or state.isLose():
            score = self.getEvaluationFunction()(state)  # type: ignore
            return score, "Stop"

        if "Stop" in legalActions:
            legalActions.remove("Stop")

        if angentIndex != state.getNumAgents() - 1:
            scores = [
                self.minValue(
                    state.generateSuccessor(angentIndex, action),
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
                    state.generateSuccessor(angentIndex, action),
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

    # End of Young's AB implementation

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor


class GigaOffensiveAgent(GigaChadAgent):
    """
    Offensive block of GigaAgent
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getFeatures(self, gameState, action):
        """
        Returns a dict of features for the state.
        The keys match up with the return from `ReflexCaptureAgent.getWeights`.
        """
        features = {}
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        capsuleList = self.getCapsules(gameState)
        hostile = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        # 1. Compute distance to the nearest food.
        if len(foodList) > 0:
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features["distanceToFood"] = minDistance
        # 2. Compute distance to the nearest capsule.
        if len(capsuleList) > 0:
            minDistance = min(
                [self.getMazeDistance(myPos, capsule) for capsule in capsuleList]
            )
            features["distanceToCapsule"] = minDistance
        # 3. Compute distance to the nearest hostile that is not scared.
        if len(hostile) > 0:
            minDistance = min(
                [
                    self.getMazeDistance(myPos, hostile.getPosition())
                    for hostile in hostile
                    if not hostile.isPacman and hostile.scaredTimer == 0
                ]
            )
            features["distanceToHostile"] = minDistance
        # 4. Compute distance to the nearest hostile that is scared.
        if len(hostile) > 0:
            minDistance = min(
                [
                    self.getMazeDistance(myPos, hostile.getPosition())
                    for hostile in hostile
                    if not hostile.isPacman and hostile.scaredTimer > 0
                ]
            )
            features["distanceToScaredHostile"] = minDistance

        features["stateScore"] = self.getScore(gameState)

        return features

    def getWeights(self, gameState, action):
        """
        Returns a dict from features to weights.
        """
        return {
            "stateScore": 100,
            "distanceToFood": -1,
            "distanceToCapsule": -1,
            "distanceToHostile": 0.5,
            "distanceToScaredHostile": -1,
        }


class GigaDefensiveAgent(GigaChadAgent):
    """
    Defensive block of GigaAgent
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

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
