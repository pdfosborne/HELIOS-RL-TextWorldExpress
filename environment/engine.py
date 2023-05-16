from textworld_express import TextWorldExpressEnv
class Engine:
    """Defines the environment function from the generator engine.
       Expects the following:
        - reset() to reset the env a start position(s)
        - step() to make an action and update the game state
        - legal_moves_generator() to generate the list of legal moves
    """
    def __init__(self, task:str='twc') -> None:
        """Initialize Engine"""
        self.Environment = TextWorldExpressEnv(envStepLimit=100)
        # Set the game generator to generate a particular game (cookingworld, twc, or coin)
        self.Environment.load(gameName=task, gameParams="numLocations=3,includeDoors=1")
        
        
    def reset(self):
        """Fully reset the environment."""
        obs, self.infos = self.Environment.reset(seed=0, gameFold="train", generateGoldPath=True)
        return obs

    
    def step(self, state:any, action:any):
        """Enact an action."""
        # In problems where the agent can choose to reset the env
        if (state=="ENV_RESET")|(action=="ENV_RESET"):
            self.reset()
            
        obs, reward, terminated, self.infos = self.Environment.step(action)
        return obs, reward, terminated

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        legal_moves = sorted(self.infos['validActions'])
        return legal_moves

