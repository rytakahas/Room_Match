import requests
import json

with open("sample_request.json") as f:
    payload = json.load(f)

res = requests.post("http://127.0.0.1:5050/room_match", json=payload)

print("Status Code:", res.status_code)
print("Response Text:")
print(res.text)

