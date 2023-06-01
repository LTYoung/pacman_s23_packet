
class OffensivePacketAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    
    """
    Offense strategy:
    1. If there are any food immediately near, eat it.
    2. If there are any food in sight, go towards it
    3. Take a move that will reduce the average distance to the farthest food and the closest food
    4. If there are any ghosts immediately near, run away
    5. If there are any ghosts in sight, take a move that will increase the distance to the closest ghost
    6. Discourage stopping
    """
    

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()
        

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            maxDistance = max([self.getMazeDistance(myPos, food) for food in foodList])
            features['minDistanceToFood'] = minDistance
            features['maxDistanceToFood'] = maxDistance
            features['avgDistanceToFood'] = (minDistance + maxDistance) / 2

        ghostList = self.getOpponents(successor)
        ghostStates = [successor.getAgentState(i) for i in ghostList]
        ghostPos = [a.getPosition() for a in ghostStates if not a.isPacman() and a.getPosition() is not None]
        if (len(ghostPos) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, ghost) for ghost in ghostPos])
            features['minDistanceToGhost'] = minDistance
            features['numGhosts'] = len(ghostPos)
        # if ghost is immediately near, run away
        if (features['minDistanceToGhost'] == 1):
            features['ghostImmediatelyNear'] = 1
        
        # if food is immediately near, eat it
        if (features['minDistanceToFood'] == 1):
            features['foodImmediatelyNear'] = 1
        else:
            features['foodImmediatelyNear'] = 0
            
        # discourage stopping
        if (action == Directions.STOP):
            features['stop'] = 1
            
        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'minDistanceToFood': -1,
            'maxDistanceToFood': 0,
            'avgDistanceToFood': 0,
            'minDistanceToGhost': 0.5,
            'numGhosts': 0,
            'foodImmediatelyNear': 0,
            'stop': -100,
            'ghostImmediatelyNear': 0
        }



class DefensivePacketAgent(ReflexCaptureAgent):
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
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2
        }
