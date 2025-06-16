#!/usr/bin/env python3
"""
学習したDeBERTa V3 Japanese tokenizerの使用例

このスクリプトは、scripts/train_deberta_v3_japanese_tokenizer.pyで学習した
トークナイザーの基本的な使用方法を示します。
"""

from transformers import DebertaV2TokenizerFast

def main():
    """学習したトークナイザーの使用例"""
    print("=== DeBERTa V3 Japanese Trained Tokenizer Example ===\n")
    
    # 学習したトークナイザーをロード
    tokenizer_path = "experiments/kuzushiji_tokenizer_deberta/deberta_v3_japanese_honkoku_hf"
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    try:
        tokenizer = DebertaV2TokenizerFast.from_pretrained(tokenizer_path)
        print("✓ Tokenizer loaded successfully!\n")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        print("Make sure you have trained the tokenizer using:")
        print("python scripts/train_deberta_v3_japanese_tokenizer.py --data_dirs data/honkoku/honkoku.txt --output_dir experiments/tokenizers --model_name deberta_v3_japanese_honkoku")
        return
    
    # トークナイザー情報表示
    print("=== Tokenizer Information ===")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"Special tokens: PAD={tokenizer.pad_token}, UNK={tokenizer.unk_token}, CLS={tokenizer.cls_token}, SEP={tokenizer.sep_token}, MASK={tokenizer.mask_token}")
    print()
    
    # 使用例のテキスト
    sample_texts = [
        "これは古典籍のテストテキストです。",
        "変体仮名やくずし字が含まれた文書の処理例。",
        "OCR技術により古文書のデジタル化が進んでいます。",
        "徳川時代の文書には難読文字が多く含まれています。",
    ]
    
    print("=== Tokenization Examples ===")
    for i, text in enumerate(sample_texts, 1):
        print(f"Example {i}: '{text}'")
        
        # トークン化
        tokens = tokenizer.tokenize(text)
        print(f"  Tokens: {tokens}")
        
        # エンコード（IDに変換）
        input_ids = tokenizer.encode(text, add_special_tokens=True)
        print(f"  Input IDs: {input_ids}")
        
        # デコード（IDからテキストに戻す）
        decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"  Decoded: '{decoded}'")
        
        # トークン数
        print(f"  Token count: {len(tokens)}")
        print()
    
    # バッチ処理の例
    print("=== Batch Processing Example ===")
    batch_encoded = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    )
    
    print(f"Batch input shape: {batch_encoded['input_ids'].shape}")
    print(f"Attention mask shape: {batch_encoded['attention_mask'].shape}")
    print("First batch item:")
    print(f"  Input IDs: {batch_encoded['input_ids'][0].tolist()}")
    print(f"  Attention mask: {batch_encoded['attention_mask'][0].tolist()}")
    
    print("\n=== Usage Complete ===")
    print("This tokenizer is now ready to be used with DeBERTa models for classical Japanese text processing!")

if __name__ == "__main__":
    main() 
