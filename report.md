# 结论

本次实验中，我们分别测试了 ChatGLM-6B、T5-3B 和 Flan-T5-3B 在 MMLU 数据集上的表现，得出其在每一个主题数据集上的回答准确率，以及平均准确率。结果见下表：

| | ChatGLM-6B | T5-3B | Flan-T5-3B|
|:---:|:---:|:---:|:---:|
|**Average Accuracy**| 0.36 | 0.25 | 0.49|

结果显示，Flan-T5-3B 的测试结果与官方一致。

T5-3B 的表现近似于随机猜测。

ChatGLM-6B 虽然参数量最大，表现却不敌 3B 参数量的 Flan-T5-3B。

# 数据集描述

MMLU 数据集内含五十余个不同主题的子数据集，每个数据集以 `.csv` 格式存储，对于每一行数据（一个 sample），均为“**问题描述,选项,答案**”的形式。

# 测试方法

## ChatGLM-6B

由于原始仓库提供了 API 脚本，且数据集本身比较合适，因此我们选择了较为直观的方式，以 API 形式实例化模型，并通过 POST 发送问题和获取回答。对于每一个子数据集：
1. 从训练集中抽取若干条数据（数量取决于参数 `--ntrain`），将其格式化为如下形式，作为 `train_prompt`：

```
problem description
choice 1
choice 2
……
Answer：A
```
2. 取出一条训练集数据，同样将其格式化为上述形式，但 Answer 一栏空置，作为 `prompt_end`。其对应真实答案作为 `lable`。将 `train_prompt` 和 `prompt_end` 拼接为 `prompt`。
3. 向 API 发送 `prompt` ，其返回作为 `response`。取其第一个字符作为回答，与 `label` 对比，从而得出模型在子数据集上的准确率。

上述方法是基于一个假设：“模型返回 `response` 的第一个字符就是模型对于给定问题的答案（选项）”。可能存在的问题在于，第一个字符可能不是模型给出的回答，但从日志观察来看，绝大部分情况下，这个假设是成立的。我们推测，这是因为 `train_prompt` 已经给出了较为格式化的样例，模型会根据其给出相同样形式的返回。且不难合理猜测，当 `ntrain` 参数越大，假设不成立的实例应当更少。

以下是一些测试日志：

### API 运行日志
```
[2023-03-29 14:58:07] ", prompt:"The following are multiple choice questions (with answers) about  abstract algebra.

Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
A. 0
B. 1
C. 2
D. 3
Answer: B

Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.
A. True, True
B. False, False
C. True, False
D. False, True
Answer: B

Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.
A. True, True
B. False, False
C. True, False
D. False, True
Answer: C

Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.
A. True, True
B. False, False
C. True, False
D. False, True
Answer: A

Find the characteristic of the ring 2Z.
A. 0
B. 3
C. 12
D. 30
Answer: A

Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.
A. 8
B. 2
C. 24
D. 120
Answer:", response:"'B'"
INFO:     127.0.0.1:43070 - "POST / HTTP/1.1" 200 OK
```

### 测试脚本运行日志

```
PROMPT: The following are multiple choice questions (with answers) about  anatomy.

What is the embryological origin of the hyoid bone?
A. The first pharyngeal arch
B. The first and second pharyngeal arches
C. The second pharyngeal arch
D. The second and third pharyngeal arches
Answer: D

Which of these branches of the trigeminal nerve contain somatic motor processes?
A. The supraorbital nerve
B. The infraorbital nerve
C. The mental nerve
D. None of the above
Answer: D

The pleura
A. have no sensory innervation.
B. are separated by a 2 mm space.
C. extend into the neck.
D. are composed of respiratory epithelium.
Answer: C

In Angle's Class II Div 2 occlusion there is
A. excess overbite of the upper lateral incisors.
B. negative overjet of the upper central incisors.
C. excess overjet of the upper lateral incisors.
D. excess overjet of the upper central incisors.
Answer: C

Which of the following is the body cavity that contains the pituitary gland?
A. Abdominal
B. Cranial
C. Pleural
D. Spinal
Answer: B

Macrostomia results from failure of fusion of
A. maxillary and mandibular processes.
B. left and right mandibular processes.
C. maxillary and frontonasal processes.
D. mandibular and hyoid arches.
Answer:
RESULT: D
```

**完整的日志文件**可以在这里获取：
- [API 日志](https://alist.arslantu.xyz/d/Document/api_chatglm.log)
- [测试脚本日志](https://alist.arslantu.xyz/d/Document/evaluate_chatglm.log)

## T5-3B 与 Flan-T5-3B

在这两个模型的评估过程中，我们使用了更严谨的方式。

`prompt` 的构造过程与前述过程相同，不同之处在于，feed prompt 后，我们取出了输出层前的 logits 层，获取选项 A B C D 对应的得分，再经过 softmax 操作，得到模型分别返回四个选项的概率，取概率最高者作为模型的回答，对比 `label` ，得到平均准确率。相关代码如下：

```python
        logits = model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids
        ).logits.flatten().float()  ## add .float()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
```