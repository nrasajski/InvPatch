import torch
import numpy as np


class ActionTokenizerBimacs:
    def __init__(self, context_len: int = 18):

        self.possible_actions = ["idle", "approach", "retreat", "lift", "place", "hold", "pour", "cut", "hammer", "saw", "stir", "screw", "drink", "wipe"]
        self.token_to_id = {val.lower(): i + 2 for i, val in enumerate(self.possible_actions)}
        self.id_to_token = {i + 2: val.lower() for i, val in enumerate(self.possible_actions)}
        self.start_token = '<BOS>'
        self.bos_token_id = 1
        self.eos_token_id = 16
        self.pad_token_id = 17
        self.pad_token = '<PAD>'
        self.end_text = '<EOS>'
        self.context_len = context_len
        self.vocab_size = len(self.possible_actions) + 3

    def encode(self, action_string):
        if isinstance(action_string, tuple) or isinstance(action_string, list):
            action_string = action_string[0]
        actions_split = action_string.split(",")
        action_ids = [self.token_to_id[action] for action in actions_split]
        tokenized_actions = [self.bos_token_id, *action_ids, self.eos_token_id]
        if len(tokenized_actions) < self.context_len:
            tokenized_actions = np.pad(np.array(tokenized_actions), (0, self.context_len - len(tokenized_actions)), 'constant')
        return torch.tensor(tokenized_actions, dtype=torch.int64).unsqueeze(dim=0)

    def get_token_to_id(self):
        return self.token_to_id

    def decode(self, tokens):
        if tokens[0] == self.bos_token_id:
            tokens = tokens[1:]
        if tokens[-1] == self.eos_token_id:
            tokens = tokens[:-1]
        text = ','.join([self.id_to_token[token] for token in tokens])
        return f"{self.start_token}{text}{self.end_text}"
