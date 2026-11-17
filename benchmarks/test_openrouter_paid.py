import json
import urllib.request
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('Thai-Temporal-Normalizer/.env', encoding='utf-8') as f:
    env = dict(line.strip().split('=', 1) for line in f if '=' in line and not line.startswith('#'))

headers = {
    'Authorization': f'Bearer {env["OPENROUTER_API_KEY"]}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://github.com/DuckerMaster/Thai-Temporal-Normalizer',
    'X-Title': 'Thai Temporal Normalizer Benchmark'
}
data = json.dumps({
    'model': 'google/gemini-2.5-flash',
    'messages': [
        {'role': 'user', 'content': 'Convert \"พรุ่งนี้\" into ISO-8601 string, assume reference time is 2026-11-08T12:00:00+07:00. Output JSON: {\"iso_value\": \"...\"}'}
    ],
    'max_tokens': 50
}).encode('utf-8')

req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', data=data, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print('OpenRouter Gemini SUCCESS Output:\n', res['choices'][0]['message']['content'].strip())
except Exception as e:
    print('OpenRouter Error:', e)
