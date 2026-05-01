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
    
class MiniLLM_RNN(nn.Module):
    """
        A Character-Level Recurrent Neural Network for next-token prediction.

        This model uses an Embedding layer followed by an RNN and a Linear layer 
        to process sequences of characters and output logits for the next character 
        at each time step (Many-to-Many architecture).

        Args:
            window (int): The length of the input sequences (context window).
            vocab_size (int): The total number of unique characters in the vocabulary.
            embedding_dim (int, optional): The size of the dense vector space for characters. Defaults to 16.
            hidden_size (int, optional): The number of features in the RNN hidden state. Defaults to 128.
            num_layers (int, optional): The number of recurrent layers. Defaults to 1.
            hidden_activation (str, optional): The non-linearity to use ('tanh' or 'relu'). Defaults to 'tanh'.
            device (str, optional): The device on which to initialize the layers ('cpu' or 'cuda'). Defaults to 'cpu'.
    """

    def __init__(self, 
                 window: int,
                 vocab_size: int,
                 embedding_dim: int = 16,
                 hidden_size: int = 128,
                 num_layers: int = 1,
                 hidden_activation: str = 'tanh',
                 device='cpu'):
        
        super(MiniLLM_RNN, self).__init__()

        self.seq_length = window
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        # building the neural network

        # embedding layer --
        # transforms integer indices into dense vectors of shape (embedding_dim)
        # output shape [batch, seq_length, embedding_dim]
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, device=self.device)

        # rnn layer --
        # transforms the embedded vector to the memory state of the rnn
        # input shape [batch, seq_length, embedding_dim]
        # output shape [batch, seq_length, hidden_size]  
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, nonlinearity=hidden_activation, batch_first=True, device=self.device)

        # fc layer --
        # apply the final linear layer to get vocabulary logits
        # input shape: (batch, seq_length, hidden_size)
        # output shape: (batch_size, seq_length, vocab_size)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size, device=self.device)
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
            Performs a forward pass of the model.

            Args:
                x (torch.Tensor): A batch of input sequences of shape (batch_size, seq_length) 
                                containing integer character indices.

            Returns:
                torch.Tensor: Flattened logits of shape (batch_size * seq_length, vocab_size) 
                            ready for CrossEntropyLoss computation.
        """
        batch_size = x.size(0)

        h0: torch.Tensor = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)

        out = self.embedding(x)
        out, _ = self.rnn(out, h0)
        out = self.fc(out)

        out = out.view(batch_size * self.seq_length, -1)

        return out