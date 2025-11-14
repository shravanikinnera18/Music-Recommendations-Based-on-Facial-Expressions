import requests

url = "http://127.0.0.1:5000/detect-emotion"
response = requests.post(url)

print(response.json())


