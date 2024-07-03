import torch


def truncate_sequence_at_first_stop_token(
    sequences: torch.Tensor, stop_tokens: torch.Tensor, fill_value: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncates sequence(s) after the first stop token and fills with `fill_value`.
    Args:
        sequences (torch.Tensor): tensor of shape [batch_size, sequence_length] or [sequence_length]
        stop_tokens (torch.Tensor): tensor containing stop tokens
        fill_value (int): value to fill the sequence with after the first stop token, usually padding ID
    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of two tensors:
            - padding_mask (torch.Tensor): tensor with the same shape as `sequences`
                where True indicates the corresponding token has been filled with `fill_value`.
            - sequences (torch.Tensor): tensor with the same shape as `sequences`
                with each sequence truncated at the first stop token and filled with `fill_value`
    """
    eos_mask = torch.isin(sequences, stop_tokens)
    seq_lens = torch.cumsum(eos_mask, dim=1)
    padding_mask = (seq_lens > 1) | ((seq_lens == 1) & ~eos_mask)
    sequences[padding_mask] = fill_value
    return padding_mask, sequences
