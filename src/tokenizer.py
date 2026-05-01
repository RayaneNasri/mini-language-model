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

    
    def encode(self, text: str, context_window=5, sequential=False) -> tuple[torch.tensor, torch.tensor]:
        """
            Prepares and encodes input and target sequences from raw text.
            This function extracts text windows to prepare data for language model training.

            Args:
                text (str): The raw text data to encode.
                context_window (int): The size of the context window (sequence length).
                sequential (bool): Indicates whether to generate sequential targets 
                                    or a single target.

            Returns:
                tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - features: The input tensor of shape (N, context_window).
                    - targets: The target tensor of shape (N,) or (N, context_window) depending on the mode.
        """

        features = []
        targets = []

        length = len(text)

        for i in range(length - context_window):

            # building the features data --
            feature = text[i:i + context_window]
            feature_encoded = [self.char_to_int[char] for char in feature]
            features.append(feature_encoded)

            # building the targets data --
            if sequential :
                target = text[i+1 : i+context_window+1]
                target_encoded = [self.char_to_int[char] for char in target]
                targets.append(target_encoded)
            else :
                targets.append( self.char_to_int[ text[i+context_window] ] )
        
        return torch.tensor(features, dtype=torch.long), torch.tensor(targets, dtype=torch.long)