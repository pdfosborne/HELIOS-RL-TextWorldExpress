from datetime import datetime
import pandas as pd
# ====== HELIOS IMPORTS =========================================
# ------ Train/Test Function Imports ----------------------------
from helios_rl import STANDARD_RL
from helios_rl import HELIOS_SEARCH
from helios_rl import HELIOS_OPTIMIZE
# ------ Config Import ------------------------------------------
# Meta parameters
from helios_rl.config import TestingSetupConfig
# Local parameters
from helios_rl.config_local import ConfigSetup
# ====== LOCAL IMPORTS ==========================================
# ------ Local Environment --------------------------------------
from environment.env import Environment
# ------ Visual Analysis -----------------------------------------------


def main():
    # ------ Load Configs -----------------------------------------
    # Meta parameters
    ExperimentConfig = TestingSetupConfig("./config.json").state_configs
    # Local Parameters
    ProblemConfig = ConfigSetup("./config_local.json").state_configs

    # Specify save dir
    version = '1.0'
    save_dir = './output/test'+'_'+version 

    num_plans = 1
    num_explor_epi = 20
    sim_threshold = 0.95

    observed_states = None
    instruction_results = None
    
    helios = HELIOS_SEARCH(Config=ExperimentConfig, LocalConfig=ProblemConfig, 
                        Environment=Environment,
                        save_dir = save_dir+'/Reinforced_Instr_Experiment',
                        num_plans = num_plans, number_exploration_episodes=num_explor_epi, sim_threshold=sim_threshold,
                        feedback_increment = 0.25, feedback_repeats=1,
                        observed_states=observed_states, instruction_results=instruction_results)

    # Don't provide any instruction information, will be defined by command line input
    helios_results = helios.search(action_cap=10, re_search_override=False, simulated_instr_goal=None)

    # Optimization phase not run on local machine
    # - Search results will save instruction complete into file directory
    # - Run man_server.py to complete training and evaluation

if __name__=='__main__':
    main()