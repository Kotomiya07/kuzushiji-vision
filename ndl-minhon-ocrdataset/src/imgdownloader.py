import glob
import os
import sys
import time

import pandas as pd
import requests
from tqdm import tqdm

if __name__ == "__main__":
    args = sys.argv
    mode = ""

    # プロキシ設定 (必要に応じて値を設定してください)
    # 例: proxies = {"http": "http://user:pass@host:port", "https": "https://user:pass@host:port"}
    proxies = {"http": "http://203.99.240.179:80", "https": "http://203.99.240.179:80"}

    # クッキー設定 (必要に応じて値を設定してください)
    # 例: cookies = {"session_id": "your_session_id_value"}
    cookies = {}

    if len(args) >= 2:
        if args[1] == "v1":
            mode = "v1"
        elif args[1] == "v2":
            mode = "v2"
        else:
            print("The parameter must be 'v1 'or 'v2'.")
            sys.exit()
    else:
        mode = "v2"
        print("The parameter must be 'v1 'or 'v2'.")
        # sys.exit()

    if mode == "v1":
        metadf = pd.read_csv("v1_metadata.csv", sep=",", dtype=str)
        metadf = metadf[~metadf["Image URL"].isna()]
        keylist = [m1 + "_" + m2 for m1, m2 in zip(metadf["Book ID"], metadf["File ID(NDL)"], strict=False)]
        key2imgurl = dict(zip(keylist, metadf["Image URL"], strict=False))

        # セッションを作成
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        session.headers.update(headers)
        # プロキシとクッキーを設定
        if proxies:
            session.proxies.update(proxies)
        if cookies:
            session.cookies.update(cookies)

        for bookdir in tqdm(glob.glob(os.path.join("v1", "*")), desc="Processing v1 books"):
            for filepath in tqdm(glob.glob(os.path.join(bookdir, "*.json")), desc="Processing v1 files"):
                bookid = os.path.basename(bookdir)
                fileid = os.path.basename(filepath).split(".")[0]
                key = bookid + "_" + fileid
                if key in key2imgurl:
                    imgurl = key2imgurl[key]
                    # 過去取得済みの画像をスキップする
                    if os.path.exists(os.path.join("img_v1", bookid, fileid + ".jpg")):
                        continue

                    try:
                        response = session.get(imgurl)
                        if response.status_code == 200:
                            rawimg = response.content
                            os.makedirs(os.path.join("img_v1", bookid), exist_ok=True)
                            with open(os.path.join("img_v1", bookid, fileid + ".jpg"), "wb") as fout:
                                fout.write(rawimg)
                            time.sleep(0.1)
                        else:
                            print(f"Error downloading {imgurl}: {response.status_code}")
                    except Exception as e:
                        print(f"Error downloading {imgurl}: {e}")
    if mode == "v2":
        metadf = pd.read_csv("v2_metadata.csv", sep="\t", dtype=str)
        metadf = metadf[~metadf["Image URL"].isna()]
        keylist = [
            m1 + "_" + m2 + "_" + m3
            for m1, m2, m3 in zip(metadf["Project ID "], metadf["Book ID"], metadf["File ID(Minna De Honkoku)"], strict=False)
        ]
        key2imgurl = dict(zip(keylist, metadf["Image URL"], strict=False))

        # セッションを作成
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
        }
        session.headers.update(headers)
        # プロキシとクッキーを設定
        if proxies:
            session.proxies.update(proxies)
        if cookies:
            session.cookies.update(cookies)

        for projectdir in tqdm(glob.glob(os.path.join("v2", "*")), desc="Processing v2 projects"):
            for bookdir in tqdm(glob.glob(os.path.join(projectdir, "*")), desc="Processing v2 books"):
                for i, filepath in enumerate(tqdm(glob.glob(os.path.join(bookdir, "*.json")), desc="Processing v2 files")):
                    projectid = os.path.basename(projectdir)
                    bookid = os.path.basename(bookdir)
                    fileid = os.path.basename(filepath).split(".")[0]
                    key = projectid + "_" + bookid + "_" + fileid
                    if key in key2imgurl:
                        imgurl = key2imgurl[key]
                        output_path = os.path.join("img_v2", projectid, bookid, fileid + ".jpg")

                        # 既存ファイルのチェック
                        if os.path.exists(output_path):
                            # print(f"Skipping {output_path} - file already exists")
                            continue

                        try:
                            # セッションを使用してリクエスト
                            # もしhttps://rmda.kulib.kyoto-u.ac.jp/にアクセスする場合はurlの最後に「?download=true」を追加する
                            if "rmda.kulib.kyoto-u.ac.jp" in imgurl:
                                imgurl = imgurl + "?download=true"
                            response = session.get(imgurl, cookies=cookies)
                            if i == 0:
                                cookies = response.cookies
                                session.cookies.update(cookies)
                            if response.status_code == 200:
                                rawimg = response.content
                                os.makedirs(os.path.join("img_v2", projectid, bookid), exist_ok=True)
                                with open(output_path, "wb") as fout:
                                    fout.write(rawimg)
                                time.sleep(0.1)
                            else:
                                print(f"Error downloading {imgurl}: {response.status_code}")
                        except Exception as e:
                            print(f"Error downloading {imgurl}: {e}")
