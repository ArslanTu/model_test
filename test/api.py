import argparse
from fastapi import FastAPI, Request
import uvicorn, json, datetime

from utils.model_get.chatglm import get_chatglm
from utils.model_get.chatglm_fintuning import get_chatglm_fintuning

app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model_selected, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    response, history = model_selected.chat(tokenizer, prompt, history=history)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer


if __name__ == '__main__':
    # read valid model list from file
    with open("models.txt", mode='r') as file:
        models = [line.strip() for line in file.readlines()]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, choices=models, required=True)
    args = parser.parse_args()
    global model, tokenizer
    model_selected = args.model
    if model_selected == "chatglm":
        model, tokenizer = get_chatglm()
    elif model_selected == "chatglm_finetuning":
        model, tokenizer = get_chatglm_fintuning()
    uvicorn.run('api:app', host='0.0.0.0', port=8000, workers=1)

model_selected = None
tokenizer = None
