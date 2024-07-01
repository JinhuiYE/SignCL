import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SignCL(nn.Module):
    def __init__(self, max_distance=32.0, pos_samples=2, neg_samples=4):
        """
        Initialize the SignCL module.

        Args:
            max_distance (float): Maximum distance to prevent negative pairs from being pushed too far.
            pos_samples (int): Number of positive samples to select.
            neg_samples (int): Number of negative samples to select.
        """
        super(SignCL, self).__init__()
        self.max_distance = max_distance
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

    def forward(self, inputs_embeds, margin=20):
        """
        Forward pass for the SignCL module.

        Args:
            inputs_embeds (torch.Tensor): Input embeddings of shape (batch_size, seq_len, embed_dim).
            margin (int): Minimum margin used for selecting negative samples.

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        batch_size, seq_len, _ = inputs_embeds.size()
        total_loss = 0

        for i in range(1, seq_len - 2):
            anchor = inputs_embeds[:, i, :].unsqueeze(1)  # Anchor sample of shape (batch_size, 1, embed_dim)

            # Positive samples selection
            pos_indices = [idx for idx in range(max(0, i - 1), min(seq_len, i + 2)) if idx != i]
            selected_pos_indices = random.sample(pos_indices, k=min(len(pos_indices), self.pos_samples))
            positives = inputs_embeds[:, selected_pos_indices, :]  # Positive samples of shape (batch_size, pos_samples, embed_dim)

            # Negative samples selection
            neg_indices = [idx for idx in range(2, seq_len - 2) if idx < i - margin or idx > i + margin]
            selected_neg_indices = random.sample(neg_indices, k=min(len(neg_indices), self.neg_samples))
            negatives = inputs_embeds[:, selected_neg_indices, :]  # Negative samples of shape (batch_size, neg_samples, embed_dim)

            # Calculate distances
            pos_dist = torch.sum(torch.abs(anchor - positives), dim=-1)  # Distance to positive samples
            neg_dist = torch.sum(torch.abs(anchor - negatives), dim=-1)  # Distance to negative samples

            # Loss calculation
            pos_loss = F.softplus(pos_dist - self.max_distance).mean()  # Positive loss
            neg_loss = F.softplus(self.max_distance - neg_dist).mean()  # Negative loss

            # Combine losses
            loss = pos_loss + neg_loss
            total_loss += loss

        # Average loss over the sequence
        total_loss /= (batch_size * (seq_len - 4)) # skip start and end frames
        return total_loss


class SignInfoNCE(nn.Module):
    def __init__(self, temperature=0.1, pos_samples=2, neg_samples=40):
        """
        Initialize the SignInfoNCE module.

        Args:
            temperature (float): Temperature scaling factor for the logits.
            pos_samples (int): Number of positive samples to select.
            neg_samples (int): Number of negative samples to select.
        """
        super(SignInfoNCE, self).__init__()
        self.temperature = temperature
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

    def forward(self, inputs_embeds, margin=20):
        """
        Forward pass for the SignInfoNCE module.

        Args:
            inputs_embeds (torch.Tensor): Input embeddings of shape (batch_size, seq_len, embed_dim).
            margin (int): Minimum margin used for selecting negative samples.

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        batch_size, seq_len, embed_dim = inputs_embeds.size()
        total_loss = 0

        for i in range(1, seq_len - 2):
            anchor = inputs_embeds[:, i, :]  # Anchor sample of shape (batch_size, embed_dim)

            # Positive samples selection
            pos_indices = [idx for idx in range(max(0, i - 1), min(seq_len, i + 2)) if idx != i]
            selected_pos_indices = random.sample(pos_indices, k=min(len(pos_indices), self.pos_samples))
            positives = inputs_embeds[:, selected_pos_indices,
                        :]  # Positive samples of shape (batch_size, pos_samples, embed_dim)

            # Negative samples selection
            neg_indices = [idx for idx in range(2, seq_len - 2) if idx < i - margin or idx > i + margin]
            selected_neg_indices = random.sample(neg_indices, k=min(len(neg_indices), self.neg_samples))
            negatives = inputs_embeds[:, selected_neg_indices,
                        :]  # Negative samples of shape (batch_size, neg_samples, embed_dim)

            # Concatenate positives and negatives
            samples = torch.cat([positives, negatives],
                                dim=1)  # Shape (batch_size, pos_samples + neg_samples, embed_dim)
            labels = torch.zeros(batch_size, len(selected_pos_indices) + len(selected_neg_indices)).to(
                inputs_embeds.device)
            labels[:, :len(selected_pos_indices)] = 1  # Positive samples labeled as 1, negative samples as 0

            # Calculate dot products and logits
            logits = torch.bmm(samples, anchor.unsqueeze(2)).squeeze(
                2) / self.temperature  # Shape (batch_size, pos_samples + neg_samples)

            # Compute InfoNCE loss
            loss = -torch.log_softmax(logits, dim=-1) * labels
            total_loss += loss.sum()

        # Average loss over the sequence
        total_loss /= (batch_size * (seq_len - 4))  # skip start and end frames
        return total_loss


if __name__ == "__main__":
    # Example usage of the SignCL class
    batch_size = 8
    seq_len = 16
    embed_dim = 64

    # Randomly generated input embeddings
    inputs_embeds = torch.randn(batch_size, seq_len, embed_dim)

    # Initialize the SignCL model
    sign_cl = SignCL(max_distance=32.0, pos_samples=2, neg_samples=4)

    # Compute the contrastive loss
    loss = sign_cl(inputs_embeds, margin=20)
    print(f"Contrastive Loss: {loss.item()}")
  
