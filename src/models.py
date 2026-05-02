# torch imports --
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# verbose -- 
from tqdm import tqdm 
import colorama
from colorama import Fore

# others --
from abc import ABC, abstractmethod

class Mlm(nn.Module, ABC):
    """
        Abstract Base Class for Character-Level Language Models.

        This class provides a common interface and shared utilities (like the 'fit' method) 
        for different architectures such as FFNN, RNN, and LSTM. It ensures that 
        all subclasses implement the required forward pass.

        Attributes:
            device (str): The device (cpu or cuda) where the model's parameters are stored.
    """

    def __init__(self, device= 'cpu'):
        super(Mlm, self).__init__()

        self.device=device

    @abstractmethod
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
            Performs a forward pass of the model.
        """
        pass

    def fit(self,
            train_loader: DataLoader, 
            test_loader: DataLoader, 
            num_epochs: int = 100,
            learning_rate: float = 0.001,
            params_dir: str = 'parameters/'
        )-> None:

        """
            Trains the neural network model and validates it periodically.

            This method implements the main training loop using CrossEntropyLoss and 
            the Adam optimizer. It includes a progress bar, accuracy tracking, and 
            performs a validation check every 10 epochs. If validation accuracy drops, 
            training is immediately stopped (strict early stopping).

            Args:
                train_loader (DataLoader): The DataLoader containing the training dataset.
                test_loader (DataLoader): The DataLoader containing the validation/test dataset.
                num_epochs (int, optional): The maximum number of epochs to train. Defaults to 100.
                learning_rate (float, optional): The learning rate for the Adam optimizer. Defaults to 0.001.
                params_dir (str, optional): The directory path where model weights (.pth) will be saved. Defaults to 'parameters/'.

            Returns:
                None
        """

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)

        total_batches = len(train_loader)

        colorama.init(autoreset=True)

        # -- Overfitting manager
        last_validation_acc = 0.0
        
        for epoch in range(num_epochs):
            self.train()

            # -- Verbose
            running_loss = 0.0
            correct_preds = 0
            total_samples = 0

            # wrap train_loader with tqdm for a progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for i, (x_batch, y_batch) in enumerate(pbar):
                x_batch, y_batch = x_batch.to(self.device), y_batch.flatten().to(self.device)
                
                # forward --
                y_hat = self(x_batch)
                loss = criterion(y_hat, y_batch)

                # backward -- 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # metrics calculation --
                running_loss += loss.item()
                
                # get the index of the highest logit (the prediction) --
                _, predicted = torch.max(y_hat, dim=1)
                correct_preds += (predicted == y_batch).sum().item()
                total_samples += y_batch.size(0)

                # update progress bar every few steps --
                if i % 10 == 0:
                    current_loss = running_loss / (i + 1)
                    current_acc = (correct_preds / total_samples) * 100
                    pbar.set_postfix({
                        'loss': f"{current_loss:.4f}", 
                        'acc': f"{current_acc:.2f}%"
                    })

            # epoch summary --
            epoch_loss = running_loss / total_batches
            epoch_acc = (correct_preds / total_samples) * 100
            print(f"\nSummary Epoch {epoch+1}: Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%\n")

            if epoch % 10 == 0 :

                # ---- validation ----
                print(Fore.YELLOW + "-- Validation --")

                self.eval()
                with torch.no_grad():

                    val_correct_preds: int = 0
                    val_total_samples: int = 0

                    for x_validation, y_validation in tqdm(test_loader) :
                        x_validation, y_validation = x_validation.to(self.device), y_validation.flatten().to(self.device)
                        y_hat = self(x_validation)
                        _, predicted = torch.max(y_hat, dim=1)
                        val_correct_preds += (y_validation == predicted).sum().item()
                        val_total_samples += y_validation.size(0)

                acc_validation = (val_correct_preds / val_total_samples) * 100

                if acc_validation > last_validation_acc :
                    print(Fore.GREEN + f"=== Saving model with a validation accuracy of {acc_validation:.4f}% ===")
                    torch.save(self.state_dict(), f"{params_dir}model_{epoch}.pth")
                    last_validation_acc = acc_validation

                else :
                    print(Fore.RED + f"=== Model has a lower validation accuracy {acc_validation:.4f}% then the last one {last_validation_acc:.4f}% ===")
                    return
                # ----------------------

        print(Fore.GREEN + f"=== Saving the final model ===")
        torch.save(self.state_dict(), f"{params_dir}model_final.pth")

class Mlm_FFNN(Mlm):
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
        
        super(Mlm_FFNN, self).__init__(device=device)
        
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
    
class Mlm_RM(nn.Module):
    """
        A Multi-Model Character-Level Language Model supporting RNN, GRU, and LSTM.

        This architecture dynamically instantiates the chosen recurrent layer type
        to process sequences of characters and output logits for next-token prediction 
        in a Many-to-Many configuration.

        Args:
            model_type (str): The type of recurrent layer to use. Must be 'rnn', 'gru', or 'lstm'.
            window (int): The length of the input sequences (context window).
            vocab_size (int): The total number of unique characters in the vocabulary.
            embedding_dim (int, optional): The dimension of the character embeddings. Defaults to 16.
            hidden_size (int, optional): The number of features in the hidden state. Defaults to 128.
            num_layers (int, optional): The number of stacked recurrent layers. Defaults to 2.
            dropout (float, optional): The dropout probability applied between recurrent layers. Defaults to 0.2.
            device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
    """

    def __init__(self,
                 model_type: str,
                 window: int,
                 vocab_size: int,
                 embedding_dim: int = 16,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 device='cpu'):
        
        super(Mlm_RM, self).__init__(device=device)

        self.model_type = model_type
        self.seq_length = window
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        rnn_classes = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }

        # building the neural network

        # embedding layer --
        # transforms integer indices into dense vectors of shape (embedding_dim)
        # output shape [batch, seq_length, embedding_dim]
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, device=self.device)

        # rnn / gru / lstm layer --
        # transforms the embedded vector to the memory state of the rnn
        # input shape [batch, seq_length, embedding_dim]
        # output shape [batch, seq_length, hidden_size]  
        self.rnn = rnn_classes[self.model_type](input_size=embedding_dim, 
                                                hidden_size=hidden_size, 
                                                num_layers=num_layers, 
                                                dropout=dropout,
                                                batch_first=True, 
                                                device=self.device)

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
        c0: torch.Tensor

        # -- embedding --
        out = self.embedding(x)
        # ---------------

        # -- rnn --
        if self.model_type == 'lstm':
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device) 
            out, _ = self.rnn(out, (h0, c0))
        else:
            out, _ = self.rnn(out, h0)
        # ---------

        # -- last layer --
        out = self.fc(out)
        # ----------------

        # flatten to match the CrossEntropyLoss format 
        out = out.view(batch_size * self.seq_length, -1)

        return out