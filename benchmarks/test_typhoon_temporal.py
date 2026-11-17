import json
import urllib.request
import sys

# Set stdout encoding
sys.stdout.reconfigure(encoding='utf-8')

with open('Thai-Temporal-Normalizer/.env', encoding='utf-8') as f:
    env = dict(line.strip().split('=', 1) for line in f if '=' in line and not line.startswith('#'))

typhoon_headers = {
    'Authorization': f'Bearer {env["TYPHOON_API_KEY"]}',
    'Content-Type': 'application/json'
}
typhoon_data = json.dumps({
    'model': 'typhoon-v2.5-30b-a3b-instruct',
    'messages': [
        {'role': 'system', 'content': 'You are a temporal extraction assistant. Convert Thai temporal expressions into ISO-8601 string.'},
        {'role': 'user', 'content': 'Convert \"พรุ่งนี้\" into ISO-8601 string, assume reference time is 2026-11-08T12:00:00+07:00. Output JSON: {\"iso_value\": \"...\"}'}
    ],
    'max_tokens': 50,
    'temperature': 0.0
}).encode('utf-8')

req1 = urllib.request.Request('https://api.opentyphoon.ai/v1/chat/completions', data=typhoon_data, headers=typhoon_headers)
try:
    with urllib.request.urlopen(req1, timeout=15) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print('Typhoon SUCCESS Output:\n', res['choices'][0]['message']['content'].strip())
except Exception as e:
    print('Typhoon Error:', e)
