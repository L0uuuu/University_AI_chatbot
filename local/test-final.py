import requests

url = "http://localhost:8000/generate"
payload = {
    "prompt": "Quels membres du personnel administratif sont des Gardiens de nuit ?",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.15
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
print(response.json())