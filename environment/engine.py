from textworld_express import TextWorldExpressEnv
import random

class Engine:
    """Defines the environment function from the generator engine.
       Expects the following:
        - reset() to reset the env a start position(s)
        - step() to make an action and update the game state
        - legal_moves_generator() to generate the list of legal moves
    """
    def __init__(self, task:str='twc', num_seeds:int=0, sub_goals:str="True") -> None:
        """Initialize Engine"""
        self.Environment = TextWorldExpressEnv(envStepLimit=100)
        # Set the game generator to generate a particular game (cookingworld, twc, or coin)
        env_parameters = {
                    'twc-easy': "numLocations=1,numItemsToPutAway=1,includeDoors=0,limitInventorySize=0",
                    'twc-medium': "numLocations=1,numItemsToPutAway=3,includeDoors=0,limitInventorySize=0",
                    'twc-hard': "numLocations=3,numItemsToPutAway=4,includeDoors=0,limitInventorySize=0",
                    'cookingworld-easy': "numLocations=1,numIngredients=1,numDistractorItems=0,includeDoors=0,limitInventorySize=0",
                    'cookingworld-medium': "numLocations=1,numIngredients=3,numDistractorItems=0,includeDoors=0,limitInventorySize=0",
                    'cookingworld-hard': "numLocations=2,numIngredients=5,numDistractorItems=0,includeDoors=0,limitInventorySize=0",
                    'coin': "numLocations=1,numDistractorItems=1,includeDoors=0,limitInventorySize=0",
                    'mapreader': "numLocations=1,maxDistanceApart=1,maxDistractorItemsPerLocation=0,includeDoors=0,limitInventorySize=0"
                    }
        # If 0 then fixed, otherwise random between limit provided
        if num_seeds==0:
            seed = 0
        else:
            seed = random.randint(0, num_seeds)
        # Define if env sub_goals are to be used
        self.sub_goals = True if sub_goals=="True" else False
        # ---
        self.Environment.load(gameName=task.split("-")[0], gameParams=env_parameters[task])
        obs, _ = self.Environment.reset(seed=seed, gameFold="train", generateGoldPath=True)
        
        # Action taken in some env to define current task
        if task.split("-")[0]=='cookingworld':
            obs, _, _, _ = self.Environment.step("read cookbook")

        print("ENV INFO")
        print("\tTask Description:", self.Environment.getTaskDescription())
        print(" ")
        print(obs)
        print(" --- ")
        print("Generation properties: " + str(self.Environment.getGenerationProperties()))
        if not self.sub_goals:
            print("Environment defined Sub-goals are DISABLED.")
        print(" --- ")
        print("Gold path: " + str(self.Environment.getGoldActionSequence()))
        print("---------------------------------------------------------")
        
    def reset(self):
        """Fully reset the environment."""
        obs, self.infos = self.Environment.reset(seed=0, gameFold="train", generateGoldPath=True)
        reward = 0
        terminated = False
        return obs, reward, terminated

    
    def step(self, state:any, action:any):
        """Enact an action."""
        # In problems where the agent can choose to reset the env
        if (state=="ENV_RESET")|(action=="ENV_RESET"):
            obs, reward, terminated = self.reset()
        else:  
            obs, reward, terminated, self.infos = self.Environment.step(action)
        
        # Engine gives reward for sub-goals
        # override this with sparse reward instead
        if not self.sub_goals:
            if (reward!=0)&(not terminated):
                print(" ")
                print("-----")
                print(action)
                print(obs)
                print("Reward from engine = ", reward)
                reward = 0
                print("Reward updated = ", reward)
            elif (reward>0)&(terminated):
                print(" ")
                print("-----")
                print(action)
                print(obs)
                print("Reward from engine = ", reward)
                reward = 1
                print("Reward updated = ", reward)
            else:
                reward = reward 
        else:
            if (reward!=0)&(not terminated):
                print(" ")
                print("-----")
                print(action)
                print(obs)
                print("Reward from engine = ", reward)
            elif (reward>0)&(terminated):
                print(" ")
                print("-----")
                print(action)
                print(obs)
                print("Reward from engine = ", reward)

        return obs, reward, terminated

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        legal_moves = sorted(self.infos['validActions'])
        return legal_moves

