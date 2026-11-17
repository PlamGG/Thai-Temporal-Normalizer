import json
import urllib.request

with open('Thai-Temporal-Normalizer/.env', encoding='utf-8') as f:
    env = dict(line.strip().split('=', 1) for line in f if '=' in line and not line.startswith('#'))

# Test Typhoon with 'typhoon-v2.5-30b-a3b-instruct'
typhoon_headers = {
    'Authorization': f'Bearer {env["TYPHOON_API_KEY"]}',
    'Content-Type': 'application/json'
}
typhoon_data = json.dumps({
    'model': 'typhoon-v2.5-30b-a3b-instruct',
    'messages': [{'role': 'user', 'content': 'สวัสดี ตอบคำเดียวว่า พร้อม'}],
    'max_tokens': 10
}).encode('utf-8')

req1 = urllib.request.Request('https://api.opentyphoon.ai/v1/chat/completions', data=typhoon_data, headers=typhoon_headers)
try:
    with urllib.request.urlopen(req1, timeout=15) as resp:
        res = json.loads(resp.read().decode())
        print('Typhoon (typhoon-v2.5-30b-a3b-instruct) SUCCESS:', res['choices'][0]['message']['content'].strip())
except Exception as e:
    print('Typhoon Error:', e)

# Test OpenRouter with 'google/gemma-4-31b-it:free'
openrouter_headers = {
    'Authorization': f'Bearer {env["OPENROUTER_API_KEY"]}',
    'Content-Type': 'application/json'
}
openrouter_data = json.dumps({
    'model': 'google/gemma-4-31b-it:free',
    'messages': [{'role': 'user', 'content': 'สวัสดี ตอบคำเดียวว่า พร้อม'}],
    'max_tokens': 10
}).encode('utf-8')

req2 = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', data=openrouter_data, headers=openrouter_headers)
try:
    with urllib.request.urlopen(req2, timeout=15) as resp:
        res = json.loads(resp.read().decode())
        print('OpenRouter (google/gemma-4-31b-it:free) SUCCESS:', res['choices'][0]['message']['content'].strip())
except Exception as e:
    print('OpenRouter Error:', e)
