import json
import requests
# local url
url = 'http://127.0.0.1:5000' # can be different

# sample data
ask = input()
data = {'text': ask
}
data = json.dumps(data)

req = requests.post(url, data)
print(req.text)
