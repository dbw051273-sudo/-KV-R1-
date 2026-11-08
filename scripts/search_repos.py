import urllib.request
import urllib.parse
import json

queries = ['VaSE KV', 'ValueDiff', 'VaSE KV cache', 'ValueDiff cache', 'VaSE reasoning']
headers = {'User-Agent': 'AntigravityResearch'}

for q in queries:
    url = f'https://api.github.com/search/repositories?q={urllib.parse.quote(q)}'
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"=== Query: {q} (Total count: {data.get('total_count', 0)}) ===")
            for item in data.get('items', [])[:4]:
                print(f"  {item['full_name']}: {item['html_url']}")
                print(f"    Description: {item.get('description')}")
    except Exception as e:
        print(f"Query {q} failed: {e}")
