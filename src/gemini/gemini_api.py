import requests
from src.config.config import GEMINI_API_KEY

GEMINI_URL = 'https://api.generativeai.google/v1beta2/models/text-bison-001:generate'

def generate_insight(prompt):
    headers = {
        'Authorization': f'Bearer {GEMINI_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "maxOutputTokens": 300
    }
    response = requests.post(GEMINI_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get('candidates', [{}])[0].get('content', 'No output')
    else:
        return f"Error: {response.status_code}, {response.text}"
