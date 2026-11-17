import json
import urllib.request
import urllib.error

def load_env():
    env = {}
    with open('Thai-Temporal-Normalizer/.env', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip()
    return env

env = load_env()

# Test Typhoon detailed
try:
    req = urllib.request.Request(
        f'{env["TYPHOON_BASE_URL"]}/models',
        headers={'Authorization': f'Bearer {env["TYPHOON_API_KEY"]}'}
    )
    with urllib.request.urlopen(req) as resp:
        models = json.loads(resp.read().decode())
        print('Typhoon Models available:', [m['id'] for m in models.get('data', [])][:5])
except urllib.error.HTTPError as e:
    print('Typhoon /models Error:', e.code, e.read().decode())
except Exception as e:
    print('Typhoon general error:', e)

# Test OpenRouter detailed
try:
    req = urllib.request.Request(
        'https://openrouter.ai/api/v1/models',
        headers={'Authorization': f'Bearer {env["OPENROUTER_API_KEY"]}'}
    )
    with urllib.request.urlopen(req) as resp:
        models = json.loads(resp.read().decode())
        free_models = [m['id'] for m in models.get('data', []) if 'free' in m['id']]
        print('OpenRouter Free Models found:', free_models[:5])
except urllib.error.HTTPError as e:
    print('OpenRouter /models Error:', e.code, e.read().decode())
except Exception as e:
    print('OpenRouter general error:', e)
