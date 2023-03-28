import argparse
import os
import numpy as np
import pandas as pd
import time
from ask import ask_chatglm_6b

choices = ["A", "B", "C", "D"]

def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval(args, subject, engine, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        # get prompt
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1] # 正确答案

        while True:
            try:
                print(f"PROMPT: {prompt}")
                response, history = ask_chatglm_6b(prompt, [])
                response = response[0]
                print(f"RESULT: {response}")
                break
            except:
                print("pausing")
                time.sleep(1)
                continue

        pred = response

        cor = pred == label
        cors.append(cor)
        all_probs = 0

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)

    result_on_cur_test = "Average accuracy {:.3f} - {}".format(acc, subject)
    with os.open(os.path.join("results", f"results_{engine}.csv"), mode='a') as file:
        file.write(f"{subject},{acc}\n")
    print(result_on_cur_test)

    return cors, acc, all_probs

def main(args):
    engines = args.engine
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for engine in engines:
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
            os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))

    print(subjects)
    print(args)

    for engine in engines:
        print(engine)
        all_cors = []
        if os.path.exists(os.path.join("results", f"results_{engine}.csv")):
            os.remove(os.path.join("results", f"results_{engine}.csv"))
        with os.open(os.path.join("results", f"results_{engine}.csv"), mode='a') as file:
            file.write("Subject,Accuracy\n")
        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

            cors, acc, probs = eval(args, subject, engine, dev_df, test_df)
            all_cors.append(cors)

            # 追加列，回答是否正确
            test_df["{}_correct".format(engine)] = cors
            test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

        weighted_acc = np.mean(np.concatenate(all_cors))
        
        result_on_cur_engine = "Average accuracy: {:.3f}".format(weighted_acc)
        with os.open(os.path.join("results", f"results_{engine}.csv"), mode='a') as file:
            file.write(f"Average,{weighted_acc}\n")
        print(result_on_cur_engine)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="../log/test_details")
    parser.add_argument("--engine", "-e", choices=["chatglm_6b"],
                        default=["chatglm_6b"], nargs="+")
    args = parser.parse_args()
    main(args)

