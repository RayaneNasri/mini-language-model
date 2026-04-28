import torch

class CharTokenizer:
    """
    A character-level tokenizer that builds a vocabulary from text and 
    encodes strings into integer tensors using a sliding context window.
    """

    def __init__(self, text: str):
        # attributes --
        self.vocabulary = sorted( list(set(text)) )
        self.char_to_int = {char: index for index, char in enumerate(self.vocabulary)}
        self.int_to_char = {index: char for char, index in self.char_to_int.items()}

    
    def encode(self, text: str, context_window=5) -> torch.Tensor:
        """
        Encodes the input text into a dataset of sliding windows.
        
        This method creates sequences of length (context_window + 1). 
        For example, if context_window=5, it returns rows of 6 integers 
        where the first 5 are the inputs (X) and the 6th is the target label (Y).

        Args:
            text (str): The raw text string to encode.
            context_window (int, optional): The number of characters used as context 
                                            to predict the next one. Defaults to 5.

        Returns:
            torch.Tensor: A 2D tensor containing the encoded text windows. 
                          Shape: (number_of_windows, context_window + 1)
        """
        data = [] # Dataset to return
        length = len(text)

        for i in range(length - context_window):
            token = text[i:i + context_window + 1]
            token_encoded = [self.char_to_int[char] for char in token]
            data.append(token_encoded)
        
        return torch.tensor(data)