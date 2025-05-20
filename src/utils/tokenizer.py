import pandas as pd
import os

# Define special token IDs as constants
PAD_ID = 0
GO_ID = 1
EOS_ID = 2  # Note: Previous version had SEP_ID=2, EOS_ID=3. Consolidating.
UNK_ID = 3  # For unknown characters encountered during encoding
# MASK_ID was 4, can be added if needed for specific pre-training tasks

# Corresponding token strings
PAD_TOKEN = '<pad>'
GO_TOKEN = '<go>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>' # For unknown characters

# Define which special tokens are part of the vocabulary by default.
# SEP_TOKEN and MASK_TOKEN can be added if their specific IDs are defined.
SPECIAL_TOKENS_LIST = [
    (PAD_ID, PAD_TOKEN),
    (GO_ID, GO_TOKEN),
    (EOS_ID, EOS_TOKEN),
    (UNK_ID, UNK_TOKEN),
]

class Vocab:
    def __init__(self, unicode_csv_path: str):
        # Initialize special token attributes directly
        self.pad_id = PAD_ID
        self.go_id = GO_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID # Store unk_id as an attribute

        self.c2i = {} # Character to ID
        self.i2c = {} # ID to Character

        # Add special tokens to mappings first
        for token_id, token_str in SPECIAL_TOKENS_LIST:
            self.c2i[token_str] = token_id
            self.i2c[token_id] = token_str
        
        next_char_id = len(SPECIAL_TOKENS_LIST)

        try:
            if not os.path.exists(unicode_csv_path):
                raise FileNotFoundError(f"Unicode CSV file not found at: {unicode_csv_path}")
            
            df = pd.read_csv(unicode_csv_path, keep_default_na=False)
            if 'char' not in df.columns:
                raise ValueError("CSV file must contain a 'char' column.")

            # Ensure characters are unique and sort them for consistency (optional but good practice)
            # Filter out any characters that might already be special token strings
            # (though unlikely if CSV contains actual textual characters)
            unique_chars = sorted(list(set(str(c) for c in df['char'].tolist() if str(c) not in self.c2i)))

            for char_val in unique_chars:
                if char_val not in self.c2i: # Redundant check if unique_chars is filtered, but safe
                    self.c2i[char_val] = next_char_id
                    self.i2c[next_char_id] = char_val
                    next_char_id += 1
            
            self.chars = "".join(unique_chars) # Store the loaded characters if needed, similar to original

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        except ValueError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading characters from CSV: {e}")
            # Depending on desired robustness, could fall back to a default small charset here,
            # but for unicode_translation.csv, it's usually critical.
            raise

    def encode(self, text_chars: str): # Expects a string of characters
        """Encodes a string of characters into a list of token IDs."""
        # Use unk_id for characters not in vocabulary
        encoded_ids = [self.go_id] + \
                      [self.c2i.get(char_val, self.unk_id) for char_val in text_chars] + \
                      [self.eos_id]
        return encoded_ids

    def decode(self, ids: list[int], join: bool = True):
        """Decodes a list of token IDs back into a string or list of characters."""
        # Filter out PAD_ID, GO_ID, EOS_ID. UNK_TOKEN will be decoded if present.
        # This behavior might need adjustment based on how one wants to see special tokens.
        decoded_chars = []
        for token_id in ids:
            if token_id not in [self.pad_id, self.go_id, self.eos_id]:
                # Get character, defaulting to UNK_TOKEN's string representation if ID is somehow unknown
                # (though all char IDs should be in i2c if unk_id is used correctly in encode)
                decoded_chars.append(self.i2c.get(token_id, UNK_TOKEN))
        
        return "".join(decoded_chars) if join else decoded_chars
        
    def __len__(self):
        """Returns the total size of the vocabulary (character set + special tokens)."""
        return len(self.c2i) # c2i now includes special tokens from the start
