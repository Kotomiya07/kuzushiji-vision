import json

import requests
from bs4 import BeautifulSoup

# 史料編纂所データベース異体字同定一覧
url = "https://wwwap.hi.u-tokyo.ac.jp/ships/itaiji_list.jsp"

# HTMLをダウンロード
response = requests.get(url)
response.encoding = "utf-8"
html = response.text

# BeautifulSoupでパース
soup = BeautifulSoup(html, "lxml")

# <table>要素を探す
table = soup.find("table")

# 各行を抽出（ヘッダーをスキップ）
data = []
for row in table.find_all("tr")[1:]:
    cols = row.find_all(["td", "th"])
    cols = [col.get_text(strip=True) for col in cols]

    if len(cols) >= 2:
        standard = cols[1]
        variants = cols[2].split() if len(cols) > 2 else []
        data.append((standard, variants))

# 確認用に数件出力
for s, vs in data[:5]:
    print(f"{s} ← {' '.join(vs)}")
    
# 出力例：
# 亜 ← 亞
# 唖 ← 啞 瘂
# 悪 ← 惡
# 芦 ← 蘆
# 鯵 ← 鰺

# 変換辞書を作成
conversion_dict = {}
for standard, variants in data:
	for variant in variants:
		conversion_dict[variant] = standard

print(conversion_dict)
# 出力例：
# {'亞': '亜', '啞': '唖', '瘂': '唖', '惡': '悪', ...}

def convert_text(text, conversion_dict):
	# 変換辞書を使ってテキストを変換
	for old, new in conversion_dict.items():
		text = text.replace(old, new)
	return text

# 使用例
converted_text = convert_text("鐡太郎 齋藤 髙橋 濵田", conversion_dict)
print(converted_text)  # 出力例： 鉄太郎 斎藤 高橋 浜田

# 変換辞書を保存
with open('conversion_dict.json', 'w', encoding='utf-8') as f:
	json.dump(conversion_dict, f, ensure_ascii=False, indent=4)
