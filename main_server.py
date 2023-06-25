from datetime import datetime
import pandas as pd
import json
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

    # ---
    # Server Specific:
    # - We do not run search methodology (assume we cannot interface directly)
    # - Search has been completed and we call the results of this to then train the agents to sub-goal/goals
    version = '1.0'
    save_dir = './output/test_'+version
    
    f = open(save_dir+'/Reinforced_Instr_Experiment/instruction_predictions.json','r')
    instruction_results = json.load(f.read())

    # ----

    # Take Instruction path now defined with reinforced+unsupervised sub-goal locations and train to these
    # Init experiment setup with sub-goal defined
    reinforced_experiment = HELIOS_OPTIMIZE(Config=ExperimentConfig, LocalConfig=ProblemConfig, 
                    Environment=Environment,
                    save_dir=save_dir+'/Reinforced_Instr_Experiment', show_figures = 'No', window_size=0.1,
                    instruction_path=None, predicted_path=instruction_results)
    reinforced_experiment.train()
    reinforced_experiment.test()
    
    # --------------------------------------------------------------------
    # Flat Baselines
    flat = STANDARD_RL(Config=ExperimentConfig, LocalConfig=ProblemConfig, 
                Environment=Environment,
                save_dir=save_dir, show_figures = 'No', window_size=0.1)
    flat.train()  
    flat.test()
    # --------------------------------------------------------------------

    

if __name__=='__main__':
    main()