from transformers import AutoTokenizer, AutoModel
import torch
import peft
from peft import LoraConfig

# pre part
class QKV_layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(QKV_layer, self).__init__()
        self.linear_q = torch.nn.Linear(in_features, out_features//3)
        self.linear_k = torch.nn.Linear(in_features, out_features//3)
        self.linear_v = torch.nn.Linear(in_features, out_features//3)

    def update(self, target_layer):
        self.linear_q.weight.data = target_layer.weight[:target_layer.out_features//3, :].data
        self.linear_q.bias.data = target_layer.bias[:target_layer.out_features//3].data

        self.linear_k.weight.data = target_layer.weight[target_layer.out_features//3:target_layer.out_features//3*2, :].data
        self.linear_k.bias.data = target_layer.bias[target_layer.out_features//3:target_layer.out_features//3*2].data

        self.linear_v.weight.data = target_layer.weight[target_layer.out_features//3*2:, :].data
        self.linear_v.bias.data = target_layer.bias[target_layer.out_features//3*2:].data
    
    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        return torch.concat([q,k,v], dim = -1)

config = LoraConfig(
    peft_type="LORA", 
    r=32, 
    lora_alpha=32, 
    target_modules=["q", "k", "v"],
    lora_dropout=0.1, 
)

# reload the model
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

# convert it again
for key, module in model.named_modules():
    if key.endswith('attention'):
        try:
            qkv_layer = QKV_layer(module.query_key_value.in_features, module.query_key_value.out_features) 
            qkv_layer.update(module.query_key_value)
            module.query_key_value = qkv_layer
        except:
            pass
        module.query_key_value = peft.tuners.lora.LoraModel(config, module.query_key_value)

# load the LoRA checkpoint
model.load_state_dict(torch.load('../../../ChatGLM-finetune-LoRA-main/saved/finetune_1.pt'), strict=False)

model.half().cuda().eval()

def get_chatglm_fintuning():
    return model, tokenizer
