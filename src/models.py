import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniLLM_FFNN(nn.Module):
    """
    A Feed-Forward Neural Network (Baseline) for character-level language modeling.
    
    This model takes a fixed-size context window of encoded characters,
    passes them through an Embedding layer, flattens the result, and 
    uses a Multi-Layer Perceptron (MLP) to predict the logits of the next character.
    """

    def __init__(self, 
                 window: int,
                 vocab_size: int,
                 embedding_dim: int = 16,
                 hidden_dim: int = 128,
                 hidden_activation = torch.tanh, 
                 device='cpu'):
        
        super(MiniLLM_FFNN, self).__init__()
        
        # building the neural network

        # embedding layer --
        # transforms integer indices into dense vectors of shape (embedding_dim)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, device=device)

        # mlp layers --
        # first linear layer takes the flattened embeddings: window * embedding_dim
        self.fc1 = nn.Linear(in_features=(window * embedding_dim), out_features=hidden_dim, device=device)
        
        # output layer maps the hidden state to the vocabulary size (raw logits)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=vocab_size, device=device)

        # the activation function applied to the hidden layer
        self.hidden_activation = hidden_activation
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): A batch of input sequences containing integer token IDs.
                              Expected shape: (batch_size, window)

        Returns:
            torch.Tensor: The raw, unnormalized scores (logits) for the next character.
                          Expected shape: (batch_size, vocab_size)
        """
        # look up the embeddings for the input characters
        # output shape: (batch_size, window, embedding_dim)
        out = self.embedding(x)
        
        # flatten the window and embedding dimensions
        # output shape: (batch_size, window * embedding_dim)
        out = out.view(out.size(0), -1)
        
        # apply the first linear layer and the activation function
        # output shape: (batch_size, hidden_dims)
        out = self.hidden_activation(self.fc1(out))
        
        # apply the final linear layer to get vocabulary logits
        # output shape: (batch_size, vocab_size)
        out = self.fc2(out)

        return out