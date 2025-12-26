import argparse
from pathlib import Path
from huggingface_hub import login
from datasets import load_dataset, Features, Image, Value

# --- 設定 ---
# 前のスクリプトでデータセットが生成されたローカルディレクトリ
LOCAL_DATASET_DIR = Path("data/line_dataset")


def main(repo_id: str, private: bool, token: str | None):
    """
    生成されたデータセットをHugging Face Hubにアップロードするメイン関数

    Args:
        repo_id: Hugging Face HubのリポジトリID (例: "your-username/kuzushiji-line-dataset")
        private: リポジトリをプライベートにするかどうか
        token: Hugging Faceのアクセストークン（指定しない場合はインタラクティブにログイン）
    """
    print("Starting dataset upload to Hugging Face Hub...")
    print(f"Local dataset directory: {LOCAL_DATASET_DIR}")
    print(f"Target repository ID: {repo_id}")

    # 1. Hugging Face Hubへのログイン
    # トークンが引数で渡された場合はそれを使用し、なければインタラクティブにログインを試みる
    try:
        if token:
            login(token=token)
            print("Logged in using the provided token.")
        else:
            # huggingface-cli login で事前にログインしていることを想定
            print("Attempting to use cached login credentials.")
            print("If login fails, please run 'huggingface-cli login' or provide a token with the --token argument.")
        
    except Exception as e:
        print(f"Login failed: {e}")
        return

    # 2. アップロードするParquetファイルのパスを収集
    data_files = {}
    for split in ["train", "val", "test"]:
        split_dir = LOCAL_DATASET_DIR / split
        if split_dir.exists():
            files = [str(p) for p in split_dir.glob("*.parquet")]
            if files:
                data_files[split] = files
            else:
                print(f"Info: No parquet files found for the '{split}' split.")
    
    if not data_files:
        print(f"Error: No data files found in {LOCAL_DATASET_DIR}. Please run the dataset creation script first.")
        return

    print(f"\nFound data files to upload: {data_files}")

    # 3. データセットを読み込む
    # datasetsライブラリは、Parquetファイル内の画像バイト列を自動的にImageオブジェクトとして解釈します。
    # Featuresを明示的に定義することで、データの型を保証します。
    try:
        features = Features({
            'image': Image(decode=True),
            'text': Value(dtype='string')
        })
        
        dataset = load_dataset(
            "parquet",
            data_files=data_files,
            features=features
        )
        print("\nSuccessfully loaded dataset from local files:")
        print(dataset)
    except Exception as e:
        print(f"\nError: Failed to load dataset from parquet files: {e}")
        return

    # 4. データセットをHubにプッシュ（アップロード）
    try:
        print(f"\nUploading dataset to '{repo_id}'...")
        # push_to_hubを実行
        dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            commit_message="Upload dataset"
        )
        print("\n✅ Dataset upload completed successfully!")
        print(f"You can find your dataset at: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please check the following:")
        print(f"- You have 'write' access to the '{repo_id}' repository.")
        print("- Your internet connection is stable.")
        print("- The repository ID is correct.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload the generated line dataset to the Hugging Face Hub.")
    parser.add_argument(
        "repo_id",
        type=str,
        help="The ID of the repository on the Hugging Face Hub (e.g., 'username/repo-name')."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If set, the repository will be created as private."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Your Hugging Face Hub access token with 'write' permission (optional)."
    )

    args = parser.parse_args()
    main(repo_id=args.repo_id, private=args.private, token=args.token)
