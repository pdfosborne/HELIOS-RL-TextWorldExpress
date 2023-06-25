from tqdm import tqdm
import time
# ------ Imports -----------------------------------------
from environment.engine import Engine
# Adapter
from adapters.default import DefaultAdapter
from adapters.language import LanguageAdapter
# Agent Setup
from helios_rl.environment_setup.imports import ImportHelper
# Evaluation standards
from helios_rl.environment_setup.results_table import ResultsTable
from helios_rl.environment_setup.helios_info import HeliosInfo

STATE_ADAPTER_TYPES = {
    "Default": DefaultAdapter,
    "Language": LanguageAdapter 
}

class Environment:

    def __init__(self, local_setup_info: dict):
        # --- INIT env from engine
        env_task = local_setup_info['env_select'] # ["cookingworld", "twc", "coin", "mapreader", "arithmetic", "sorting"]
        self.env = Engine(task=env_task)
        self.start_obs,_,_ = self.env.reset()
        # ---
        # --- PRESET HELIOS INFO
        # Agent
        Imports = ImportHelper(local_setup_info)
        self.agent, self.agent_type, self.agent_name, self.agent_state_adapter = Imports.agent_info(STATE_ADAPTER_TYPES)
        self.num_train_episodes, self.num_test_episodes, self.training_action_cap, self.testing_action_cap, self.reward_signal = Imports.parameter_info()  
        # Training or testing phase flag
        self.train = Imports.training_flag()
        # --- HELIOS
        self.live_env, self.observed_states, self.experience_sampling = Imports.live_env_flag()
        # Results formatting
        self.results = ResultsTable(local_setup_info)
        # HELIOS input function
        # - We only want to init trackers on first batch otherwise it resets knowledge
        self.helios = HeliosInfo(self.observed_states, self.experience_sampling)
        # Env start position for instr input
        # Enable sub-goals
        if (local_setup_info['sub_goal'] is not None) & (local_setup_info['sub_goal']!=["None"]) & (local_setup_info['sub_goal']!="None"):
            self.sub_goal:list = local_setup_info['sub_goal']
        else:
            self.sub_goal:list = None

    def episode_loop(self):
        # Mode selection (already initialized)
        if self.train:
            number_episodes = self.num_train_episodes
        else:
            number_episodes = self.num_test_episodes

        for episode in tqdm(range(0, number_episodes)):
            action_history = []
            # ---
            # Start observation is used instead of .reset() fn so that this can be overriden for repeat analysis from the same start pos
            obs,_,_ = self.env.reset()
            legal_moves = self.env.legal_move_generator(obs)
            state = self.agent_state_adapter.adapter(state=obs, legal_moves=legal_moves, episode_action_history=action_history, encode=True)
            # ---
            start_time = time.time()
            episode_reward:int = 0
            for action in range(0,self.training_action_cap+1):
                if self.live_env:
                    # Agent takes action
                    legal_moves = self.env.legal_move_generator(obs)
                    agent_action = self.agent.policy(state, legal_moves)
                    
                    
                    
                    next_obs, reward, terminated = self.env.step(state=obs, action=agent_action)
                    # Can override reward per action with small negative punishment
                    if reward==0:
                        reward = self.reward_signal[1]
                    # Custom reward to discourage agent getting stuck
                    if agent_action in action_history:
                        reward = self.reward_signal[3]
                    action_history.append(agent_action)
                    
                    legal_moves = self.env.legal_move_generator(next_obs) 
                    next_state = self.agent_state_adapter.adapter(state=next_obs, legal_moves=legal_moves, episode_action_history=action_history, encode=True)
                    # HELIOS trackers    
                    self.helios.observed_state_tracker(engine_observation=next_obs,
                                                        language_state=self.agent_state_adapter.adapter(state=next_obs, legal_moves=legal_moves, episode_action_history=action_history, encode=False))
                    
                    # MUST COME BEFORE SUB-GOAL CHECK OR 'TERMINAL STATES' WILL BE FALSE
                    self.helios.experience_sampling_add(state, agent_action, next_state, reward, terminated)
                    # Trigger end on sub-goal if defined
                    if self.sub_goal:
                        if (type(self.sub_goal)==type(''))|(type(self.sub_goal)==type(0)):
                            if next_obs == self.sub_goal:
                                reward = self.reward_signal[0]
                                terminated = True
                        elif (type(self.sub_goal)==type(list('')))|(type(self.sub_goal)==type([0])):    
                            if next_obs in self.sub_goal:
                                reward = self.reward_signal[0]
                                terminated = True         
                        else:
                            print("Sub-Goal Type ERROR: The input sub-goal type must be a str/int or list(str/int).")               
                else:
                    # Experience Sampling
                    legal_moves = self.helios.experience_sampling_legal_actions(state)
                    # Unknown state, have no experience to sample from so force break episode
                    if legal_moves == None:
                        break
                    
                    agent_action = self.agent.policy(state, legal_moves)
                    next_state, reward, terminated = self.helios.experience_sampling_step(state, agent_action)

                # If action limit reached
                # - include learning and episode reward output if needed
                if (action >= self.training_action_cap)&(not terminated):
                    reward = self.reward_signal[2] 
                    terminated = True    
                episode_reward+=reward 
                
                if self.train:
                    self.agent.learn(state, next_state, reward, agent_action)

                if terminated:
                    break
                else:    
                    state=next_state
                    if self.live_env:
                        obs = next_obs        
            
            end_time = time.time()
            agent_results = self.agent.q_result()
            if self.live_env:
                self.results.results_per_episode(self.agent_name, None, episode, action, episode_reward, (end_time-start_time), action_history, agent_results[0], agent_results[1]) 

        return self.results.results_table_format()
                    
