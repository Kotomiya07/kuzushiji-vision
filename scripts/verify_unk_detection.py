#!/usr/bin/env python3
"""
UNK検出の正確性を検証するスクリプト
"""

from transformers import AutoTokenizer


def verify_unk_detection():
    """UNK検出の正確性を検証"""
    print("=== UNK検出の検証 ===\n")

    # トークナイザーをロード
    tokenizer = AutoTokenizer.from_pretrained("globis-university/deberta-v3-japanese-xsmall")
    unk_token_id = tokenizer.convert_tokens_to_ids("[UNK]")
    print(f"UNKトークンID: {unk_token_id}")
    
    # 実際にUNKを含む文を調査
    test_sentences = [
        "て阿蘇噴火脈の蹝を垂るゝのみ美濃尾張近江の三国には従来の調査に拠れば一箇の",
        "夏莱菔は真盛り、二",
        "此レヲシテ豊饒ナラシメ又早晹ノ雰囲気中ニ配分シテ之",
        "バー」ノ若干湖及𤇆煙ヲ発出スル尖円ノ小群山ヲ見ル",
    ]

    print("=== 各文の詳細分析 ===")
    for i, sentence in enumerate(test_sentences):
        print(f"\n文 {i + 1}: {sentence}")
        
        # トークン化
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # UNKを探す
        unk_positions = [j for j, token_id in enumerate(token_ids) if token_id == unk_token_id]
        unk_token_positions = [j for j, token in enumerate(tokens) if token == "[UNK]"]
        
        print(f"  トークン: {tokens}")
        print(f"  トークンID: {token_ids}")
        print(f"  UNKトークン位置: {unk_token_positions}")
        print(f"  UNKトークンID位置: {unk_positions}")
        
        # 復元して比較
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"  復元文: {decoded}")
        print(f"  元の文と一致: {sentence.strip() == decoded.strip()}")
        
        # 文字単位で比較
        if sentence.strip() != decoded.strip():
            print("  差異分析:")
            for j, (orig, dec) in enumerate(zip(sentence, decoded)):
                if orig != dec:
                    print(f"    位置 {j}: '{orig}' → '{dec}'")
    
    # 個別文字のテスト
    print("\n=== 個別文字のUNKテスト ===")
    problematic_chars = ["蹝", "菔", "晹", "𤇆", "㐫", "℥", "◓", "㑹", "㔫", "㕝"]
    
    for char in problematic_chars:
        # 単体文字のテスト
        tokens = tokenizer.tokenize(char)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        print(f"文字 '{char}' (U+{ord(char):04X}):")
        print(f"  トークン: {tokens}")
        print(f"  トークンID: {token_ids}")
        print(f"  UNK含有: {'[UNK]' in tokens}")
        print(f"  復元: '{decoded}'")
        print(f"  一致: {char == decoded}")
        
        # エンコード->デコードの詳細チェック
        encoded = tokenizer.encode(char, add_special_tokens=False)
        decoded_from_ids = tokenizer.decode(encoded)
        print(f"  エンコードIDs: {encoded}")
        print(f"  IDからの復元: '{decoded_from_ids}'")
        
        # UNKトークンIDが含まれているかチェック
        unk_in_ids = unk_token_id in encoded
        print(f"  UNKトークンID含有: {unk_in_ids}")
        print()

    # 元のテキストファイルから実際にUNKを含む文を探す
    print("=== 実際のUNKを含む文の検索 ===")
    try:
        with open("data/honkoku/honkoku.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        unk_containing_sentences = []
        for i, line in enumerate(lines[:1000]):  # 最初の1000行のみチェック
            line = line.strip()
            if not line:
                continue
                
            # トークン化してUNKをチェック
            tokens = tokenizer.tokenize(line)
            if "[UNK]" in tokens:
                unk_containing_sentences.append((i, line, tokens))
                if len(unk_containing_sentences) >= 5:  # 最初の5文だけ
                    break
        
        if unk_containing_sentences:
            print(f"実際にUNKを含む文を {len(unk_containing_sentences)} 個発見:")
            for line_num, sentence, tokens in unk_containing_sentences:
                print(f"\n行 {line_num + 1}: {sentence}")
                print(f"  トークン: {tokens}")
                
                # UNKの位置を特定
                unk_positions = [j for j, token in enumerate(tokens) if token == "[UNK]"]
                print(f"  UNK位置: {unk_positions}")
                
                # 復元して差異を確認
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                print(f"  復元: {decoded}")
                print(f"  一致: {sentence == decoded}")
        else:
            print("最初の1000行でUNKを含む文は見つかりませんでした")
            
    except FileNotFoundError:
        print("テキストファイルが見つかりません")


if __name__ == "__main__":
    verify_unk_detection() 
