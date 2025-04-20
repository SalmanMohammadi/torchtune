import torch
import torch.nn as nn
from torchtune.rlhf import ChosenRejectedOutputs

class ChunkedDPOLoss(nn.Module):
    def __init__(
        self,
        num_output_chunks: int = 8,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.beta = beta
        self.label_smoothing = label_smoothing

    def calculate_chunked_loss(
        self,   
        weight,
        hidden_states,
        ref_weight,
        ref_hidden_states,
    ):
        hidden_states = hidden_states.chunk(self.num_output_chunks, dim=1)
        ref_hidden_states = ref_hidden_states.chunk(self.num_output_chunks, dim=1)

    )
    def forward(
            self,
            inputs: torch.Tensor,
            ref_inputs: torch.Tensor,
            weight,
            hidden_states,
            ref_weight,
            ref_hidden_states,
    ):
        len_chosen = inputs.shape[0] // 2
        chosen_inputs = torch.chunk(inputs[:len])
        
        hidden_states = hidden_states.chunk(self.num_output_chunks, dim=1)
        ref_hidden_states = ref_hidden_states.chunk(self.num_output_chunks, dim=1)

        total_loss = 0
        for (hidden_state, ref_hidden_state) in zip(hidden_states, ref_hidden_states):
            loss = self.compute_loss(weight, hidden_state, ref_weight, ref_hidden_state)
            total_loss += loss
        return total_loss / self.num_output_chunks

    def compute_loss(self, weight, hidden_state, ref_weight, ref_hidden_state):
        