# %%
from transformers import AutoTokenizer

tokenizer_unigram = AutoTokenizer.from_pretrained("experiments/kuzushiji_tokenizer_hf/vocab32000_unigram")
tokenizer_bpe = AutoTokenizer.from_pretrained("experiments/kuzushiji_tokenizer_hf/vocab32000_bpe")

print("Unigram Tokenizer:")
print(tokenizer_unigram.encode("いろはにほへとちりぬるを"))
print(tokenizer_unigram.convert_ids_to_tokens(tokenizer_unigram.encode("いろはにほへとちりぬるを")))

print("BPE Tokenizer:")
print(tokenizer_bpe.encode("いろはにほへとちりぬるを"))
print(tokenizer_bpe.convert_ids_to_tokens(tokenizer_bpe.encode("いろはにほへとちりぬるを")))

# %%
