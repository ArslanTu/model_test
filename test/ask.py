import requests
import json

def ask_chatglm_6b(prompt, history):
    url = "http://127.0.0.1:8000"
    data = {"prompt": prompt, "history": history}
    res = requests.post(url, json.dumps(data)).json()
    return res.get("response"), res.get("history")