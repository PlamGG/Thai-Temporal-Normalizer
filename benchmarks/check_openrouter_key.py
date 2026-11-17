import json
import urllib.request

with open('Thai-Temporal-Normalizer/.env', encoding='utf-8') as f:
    env = dict(line.strip().split('=', 1) for line in f if '=' in line and not line.startswith('#'))

headers = {'Authorization': f'Bearer {env["OPENROUTER_API_KEY"]}'}
req = urllib.request.Request('https://openrouter.ai/api/v1/auth/key', headers=headers)
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print("OpenRouter Key Info:", data)
except Exception as e:
    print("OpenRouter key error:", e)
