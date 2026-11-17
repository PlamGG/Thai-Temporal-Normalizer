import json
import urllib.request
import urllib.error

with open('Thai-Temporal-Normalizer/.env', encoding='utf-8') as f:
    env = dict(line.strip().split('=', 1) for line in f if '=' in line and not line.startswith('#'))

# Check Typhoon raw models
req = urllib.request.Request(
    'https://api.opentyphoon.ai/v1/models',
    headers={'Authorization': f'Bearer {env["TYPHOON_API_KEY"]}'}
)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    print("Typhoon models returned:", data)

# Search OpenRouter for meta-llama or gemini free models
req2 = urllib.request.Request(
    'https://openrouter.ai/api/v1/models',
    headers={'Authorization': f'Bearer {env["OPENROUTER_API_KEY"]}'}
)
with urllib.request.urlopen(req2) as resp:
    data2 = json.loads(resp.read().decode())
    models2 = [m['id'] for m in data2.get('data', [])]
    llama_free = [m for m in models2 if 'llama' in m.lower() and 'free' in m.lower()]
    gemini_free = [m for m in models2 if 'gemini' in m.lower() and 'free' in m.lower()]
    print("Llama free:", llama_free)
    print("Gemini free:", gemini_free)
