from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor
# StateAdapter includes static methods for adapters
from helios_rl.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

class LanguageAdapter:
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self):
        # Language encoder doesn't require any preset knowledge of env to use
        self.encoder = LanguageEncoder()
    
    def adapter(self, state:any, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use Language name for every piece name for current board position """
        if len(episode_action_history)>0:
            state = state + ' '+ episode_action_history[-1] + '.'
        # Encode to Tensor for agents
        if encode:
            state_encoded = self.encoder.encode(state=state)
        else:
            state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in LanguageAdapter._cached_state_idx):
                    LanguageAdapter._cached_state_idx[sent] = len(LanguageAdapter._cached_state_idx)
                state_indexed.append(LanguageAdapter._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)

        return state_encoded