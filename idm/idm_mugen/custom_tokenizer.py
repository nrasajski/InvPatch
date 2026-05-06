import torch
import numpy as np

class ActionTokenizerMUGEN:
    def __init__(self, context_len: int = 120):

        self.possible_actions = ["inaction", "move_right", "move_left", "jump", "jump_right", "jump_left", "descend"]
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.delimiter_token = '<DELIMITER>'
        self.eos_token = '<EOS>'

        self.padding_token_id = 0
        self.bos_token_id = 1
        self.delimiter_token_id = len(self.possible_actions) + 2
        self.eos_token_id = self.delimiter_token_id + 1

        self.token_to_id = {val.lower(): i + 2 for i, val in enumerate(self.possible_actions)}
        self.id_to_token = {i + 2: val.lower() for i, val in enumerate(self.possible_actions)}

        self.id_to_token[self.padding_token_id] = self.pad_token
        self.token_to_id[self.pad_token] = self.padding_token_id

        self.id_to_token[self.bos_token_id] = self.bos_token
        self.token_to_id[self.bos_token] = self.bos_token_id

        self.id_to_token[self.delimiter_token_id] = self.delimiter_token
        self.token_to_id[self.delimiter_token] = self.delimiter_token_id

        self.id_to_token[self.eos_token_id] = self.eos_token
        self.token_to_id[self.eos_token] = self.eos_token_id

        print(self.token_to_id)

        self.context_len = context_len
        self.vocab_size = len(self.possible_actions) + 4

    def encode(self, action_string):
        if isinstance(action_string, tuple) or isinstance(action_string, list):
            action_string = action_string[0]
        actions_split = action_string.split(",")
        action_ids = [self.token_to_id[action] for action in actions_split]
        tokenized_actions = [self.bos_token_id, *action_ids, self.eos_token_id]
        padded_actions = np.pad(np.array(tokenized_actions), (self.padding_token_id, self.context_len - len(tokenized_actions)), 'constant')
        return torch.tensor(padded_actions, dtype=torch.int64).unsqueeze(dim=0)

    def get_token_to_id(self):
        return self.token_to_id

    def decode(self, tokens):
        if tokens[0] == self.bos_token_id:
            tokens = tokens[1:]
        if tokens[-1] == self.eos_token_id:
            tokens = tokens[:-1]
        text = ','.join([self.id_to_token[token] for token in tokens])
        return f"{self.bos_token}{text}{self.eos_token}"
