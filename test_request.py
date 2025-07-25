import requests

# Change this if running remotely
url = "https://sematic-similarity.onrender.com/predict"

# Sample request payload
data = {
    "text1": "India is a democratic country.",
    "text2": "India follows a system of democracy."
}

# Make POST request
response = requests.post(url, json=data)

# Print response
if response.status_code == 200:
    print("✅ Similarity Score:", response.json()["similarity score"])
else:
    print("❌ Error:", response.status_code, response.text)
