import json
import os
import urllib.request

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

# Test Typhoon
typhoon_headers = {
    'Authorization': f'Bearer {env["TYPHOON_API_KEY"]}',
    'Content-Type': 'application/json'
}
typhoon_data = json.dumps({
    'model': env.get('TYPHOON_MODEL', 'typhoon-v1.5x-70b-instruct'),
    'messages': [{'role': 'user', 'content': 'สวัสดี ตอบคำเดียวว่า พร้อม'}],
    'max_tokens': 15
}).encode('utf-8')

req1 = urllib.request.Request(f'{env["TYPHOON_BASE_URL"]}/chat/completions', data=typhoon_data, headers=typhoon_headers)
try:
    with urllib.request.urlopen(req1, timeout=15) as resp:
        res = json.loads(resp.read().decode())
        print('Typhoon OK:', res['choices'][0]['message']['content'].strip())
except Exception as e:
    print('Typhoon Error:', e)

# Test OpenRouter
openrouter_headers = {
    'Authorization': f'Bearer {env["OPENROUTER_API_KEY"]}',
    'Content-Type': 'application/json'
}
openrouter_data = json.dumps({
    'model': env.get('OPENROUTER_MODEL', 'google/gemini-2.0-flash-exp:free'),
    'messages': [{'role': 'user', 'content': 'Hello, reply Ready.'}],
    'max_tokens': 15
}).encode('utf-8')

req2 = urllib.request.Request(f'{env["OPENROUTER_BASE_URL"]}/chat/completions', data=openrouter_data, headers=openrouter_headers)
try:
    with urllib.request.urlopen(req2, timeout=15) as resp:
        res = json.loads(resp.read().decode())
        print('OpenRouter OK:', res['choices'][0]['message']['content'].strip())
except Exception as e:
    print('OpenRouter Error:', e)
