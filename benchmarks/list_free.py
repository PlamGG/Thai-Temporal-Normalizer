import json
import urllib.request

with open('Thai-Temporal-Normalizer/.env', encoding='utf-8') as f:
    env = dict(line.strip().split('=', 1) for line in f if '=' in line and not line.startswith('#'))

req = urllib.request.Request('https://openrouter.ai/api/v1/models', headers={'Authorization': f'Bearer {env["OPENROUTER_API_KEY"]}'})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    models = data.get('data', [])
    free = [m['id'] for m in models if m.get('pricing', {}).get('prompt') == '0' or ':free' in m['id']]
    print('Total free models on OpenRouter:', len(free))
    for m in free:
        if any(keyword in m.lower() for keyword in ['qwen', 'deepseek', 'gemma', 'phi', 'mistral', 'llama', 'gemini']):
            print('Notable free model:', m)
