import pathlib
import os # pathlibが使えない古いPython環境用。基本はpathlibを推奨

def analyze_text_file_lengths(target_directory_str):
    """
    指定されたディレクトリ以下のテキストファイルの文字数を計測し、
    最大長と平均長を出力します。
    """
    target_dir = pathlib.Path(target_directory_str)

    if not target_dir.is_dir():
        print(f"エラー: ディレクトリ '{target_dir}' が見つかりません。")
        return

    all_lengths = []
    processed_files_count = 0

    print(f"ディレクトリ '{target_dir}' 内のファイルを検索・処理しています...")

    # target_dir以下のすべてのファイルを再帰的に検索
    # pathlib.Path.rglob('*') はディレクトリも含むので、is_file()でフィルタリング
    for file_path in target_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix == '.txt':
            try:
                # ファイルをUTF-8として読み込み
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 文字数を取得
                # ここでいう「文の文字数」はファイル全体の文字数と解釈します。
                # もしファイル内の個々の文を区切って処理する場合は、
                # sentence tokenization (文分割) の処理が必要になります。
                current_length = len(content)
                if current_length > 100:
                    print(f"  処理中: {file_path} - 文字数: {current_length}") # 詳細ログ
                all_lengths.append(current_length)
                processed_files_count += 1
                # print(f"  処理中: {file_path.name} - 文字数: {current_length}") # 詳細ログ
            except UnicodeDecodeError:
                print(f"  警告: ファイル '{file_path}' はUTF-8でデコードできませんでした。スキップします。")
            except Exception as e:
                print(f"  警告: ファイル '{file_path}' の処理中にエラーが発生しました: {e}。スキップします。")

    if not all_lengths:
        print("処理対象のファイルが見つかりませんでした。")
        return

    max_length = max(all_lengths)
    # 平均を計算する際、リストが空でないことを確認済みなのでゼロ除算の心配はない
    average_length = sum(all_lengths) / len(all_lengths)

    print(f"\n--- 解析結果: {target_directory_str} ---")
    print(f"処理されたファイル数: {processed_files_count}")
    print(f"最大の文の長さ（文字数）: {max_length}")
    print(f"平均の文の長さ（文字数）: {average_length:.2f}") # 小数点以下2桁で表示

if __name__ == "__main__":
    # スクリプトを実行する場所からの相対パス、または絶対パスを指定
    # 例: スクリプトが 'my_scripts' ディレクトリにあり、
    # 'my_scripts' と 'kokubunken_repo' が同じ階層にある場合:
    # path_to_texts = "../kokubunken_repo/text"
    #
    # スクリプトが 'kokubunken_repo' の親ディレクトリにある場合:
    # path_to_texts = "kokubunken_repo/text"
    #
    # このスクリプトファイル自体が kokubunken_repo/text のどこかにある場合は
    # より複雑な相対パス指定が必要になるか、絶対パスを使用します。
    # ここでは、カレントワーキングディレクトリからの相対パスとします。
    
    target_repo_paths = [
        "ndl-minhon-ocrdataset/src/honkoku_oneline_v1",
        "ndl-minhon-ocrdataset/src/honkoku_oneline_v2",
        "honkoku_yatanavi/honkoku_oneline",
        "data/oneline",
        "kokubunken_repo/text",
    ]

    for target_repo_path in target_repo_paths:
        analyze_text_file_lengths(target_repo_path)
