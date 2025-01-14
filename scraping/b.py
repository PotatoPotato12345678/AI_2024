import requests
import xml.etree.ElementTree as ET

# ArXiv APIのエンドポイント
base_url = "http://export.arxiv.org/api/query"

# 分野ごとの結果を格納する辞書
category_counts = {}

# カテゴリリスト（例としていくつかのカテゴリ）
categories = ["astro-ph"]

for category in categories:
    query = f"cat:{category}"
    
    params = {
        "search_query": query,
        "max_results": 0,  # 結果数のみ取得
    }
    
    response = requests.get(base_url, params=params)
    root = ET.fromstring(response.text)
    
    # 総結果数を取得
    total_results = int(root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults").text)
    category_counts[category] = total_results

# 結果を表示
for category, count in category_counts.items():
    print(f"{category}: {count} papers")