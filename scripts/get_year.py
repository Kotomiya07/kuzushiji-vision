import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# 書籍データのリスト
books_data = [
    {"bookid": "100241706", "書籍名": "虚南留別志"},
    {"bookid": "100249371", "書籍名": "鼎左秘録"},
    {"bookid": "100249376", "書籍名": "御前菓子秘伝抄"},
    {"bookid": "100249416", "書籍名": "餅菓子即席／手製集"},
    {"bookid": "100249476", "書籍名": "飯百珍伝"},
    {"bookid": "100249537", "書籍名": "料理珍味集"},
    {"bookid": "200003076", "書籍名": "好色一代男"},
    {"bookid": "200003803", "書籍名": "源氏物語"},
    {"bookid": "200003967", "書籍名": "おらが春"},
    {"bookid": "200004107", "書籍名": "二人比丘尼"},
    {"bookid": "200004148", "書籍名": "椿説弓張月"},
    {"bookid": "200005598", "書籍名": "傾城買四十八手"},
    {"bookid": "200005798", "書籍名": "世間胸算用／大晦日ハ一日千金"},
    {"bookid": "200006663", "書籍名": "ぢぐち"},
    {"bookid": "200006665", "書籍名": "吉利支丹物語"},
    {"bookid": "200008003", "書籍名": "歌学提要"},
    {"bookid": "200008316", "書籍名": "武家義理物語"},
    {"bookid": "200010454", "書籍名": "源氏物語"},
    {"bookid": "200014685", "書籍名": "南総里見八犬伝"},
    {"bookid": "200014740", "書籍名": "雨月物語"},
    {"bookid": "200015779", "書籍名": "浮世風呂"},
    {"bookid": "200015843", "書籍名": "日本永代蔵"},
    {"bookid": "200017458", "書籍名": "曾我物語"},
    {"bookid": "200018243", "書籍名": "玉くしげ"},
    {"bookid": "200019865", "書籍名": "女郎花物語"},
    {"bookid": "200020019", "書籍名": "竹斎"},
    {"bookid": "200021063", "書籍名": "うすゆき物語"},
    {"bookid": "200021071", "書籍名": "伊曾保物語"},
    {"bookid": "200021086", "書籍名": "伊曾保物語"},
    {"bookid": "200021637", "書籍名": "当世料理"},
    {"bookid": "200021644", "書籍名": "菓子話船橋"},
    {"bookid": "200021660", "書籍名": "養蚕秘録"},
    {"bookid": "200021712", "書籍名": "万宝料理秘密箱"},
    {"bookid": "200021763", "書籍名": "膳部料理抄"},
    {"bookid": "200021802", "書籍名": "料理物語"},
    {"bookid": "200021851", "書籍名": "かてもの"},
    {"bookid": "200021853", "書籍名": "日用惣菜俎不時珍客即席庖丁"},
    {"bookid": "200021869", "書籍名": "料理方心得之事"},
    {"bookid": "200021925", "書籍名": "新編異国料理"},
    {"bookid": "200022050", "書籍名": "料理秘伝抄"},
    {"bookid": "200025191", "書籍名": "仁勢物語"},
    {"bookid": "brsk00000", "書籍名": "物類称呼"},
    {"bookid": "hnsd00000", "書籍名": "比翼連理花迺志満台"},
    {"bookid": "umgy00000", "書籍名": "春色梅児与美"},
]

# ベースURLのフォーマット
base_url = "https://kokusho.nijl.ac.jp/biblio/{}/1?ln=ja"

# 成立年のXPath (修正版)
# /html/body/div/div[2]/div[11]/div[2] このパス以下で、
# 「成立年」というテキストを直接持つdiv要素を探し、
# そのdiv要素の直後にある最初のdiv兄弟要素を取得します。
# これにより、特定のdivのインデックスに依存せず、柔軟に成立年を見つけることができます。
xpath_成立年 = "/html/body/div/div[2]/div[11]/div[2]//div[text()='成立年']/following-sibling::div[1]"


# --- Selenium セットアップ ---
# Chromeオプションを設定（ヘッドレスモードでGUIなしで実行）
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # ブラウザのGUIを表示しない
options.add_argument("--disable-gpu") # ヘッドレスモードでの推奨設定
options.add_argument("--no-sandbox") # 一部の環境で必要
options.add_argument("--disable-dev-shm-usage") # リソース不足の問題を回避
options.add_argument("--window-size=1920,1080") # ウィンドウサイズを設定

# ChromeDriverのパスを指定 (PATHが通っている場合は不要)
# service = Service(executable_path='/path/to/your/chromedriver')
# driver = webdriver.Chrome(service=service, options=options)

driver = None # 初期化

try:
    # WebDriverを初期化 (ChromeDriverがPATHにある場合)
    driver = webdriver.Chrome(options=options)
    # ページロードのタイムアウトを設定
    driver.set_page_load_timeout(30)

    # ヘッダーの出力
    print("bookid\t書籍名\t成立年")

    # 各書籍について情報を取得
    for book in books_data:
        bookid = book["bookid"]
        book_name = book["書籍名"]
        url = base_url.format(bookid) # URLを構築
        成立年 = "N/A" # デフォルト値

        try:
            # URLへアクセス
            driver.get(url)

            # 修正されたXPathで要素が見つかるまで待機（最大20秒）
            # JavaScriptによるコンテンツの読み込みを待つため、Requestsより長めのタイムアウトを設定
            wait = WebDriverWait(driver, 20)
            成立年_element = wait.until(
                EC.presence_of_element_located((By.XPATH, xpath_成立年))
            )

            # 要素のテキストコンテンツを取得し、余分な空白を除去
            成立年 = 成立年_element.text.strip()

        except TimeoutException:
            成立年 = "タイムアウト: 成立年要素が指定時間内に見つかりませんでした。"
        except WebDriverException as e:
            成立年 = f"WebDriverエラー: {e}"
        except Exception as e:
            成立年 = f"予期せぬエラー: {e}"

        # 結果を出力
        print(f"### {bookid}\t{book_name}\n時代: {成立年}\n文体: \n")

        # サーバーへの負荷を考慮し、リクエスト間に1秒の待機時間を設ける
        time.sleep(1)

except WebDriverException as e:
    print(f"WebDriverの初期化エラー: {e}")
    print("ChromeDriverが正しくインストールされ、PATHが通っているか確認してください。")
    print("または、Service(executable_path='/path/to/your/chromedriver') を使用してパスを明示的に指定してください。")
finally:
    # エラーが発生してもブラウザが確実に閉じられるようにする
    if driver:
        driver.quit()
