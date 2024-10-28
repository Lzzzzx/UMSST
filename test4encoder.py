# 载入预训练部分的encoder 与 mlp参数
import torch
from collections import OrderedDict
from copy import deepcopy

# 载入预训练模型
arg_encoder_total=torch.load('./experiments/NYCBike1/20230829-063418/best_encoder_model.pth')['model']
key_to_mlp=list(arg_encoder_total.keys())[-4:]
key_to_encoder=list(arg_encoder_total.keys())[:-4]
# 获取encoder部分模型参数
arg_encoder=deepcopy(arg_encoder_total)
for key in key_to_mlp:
    del arg_encoder[key]
# 获取mlp部分模型参数
arg_mlp=deepcopy(arg_encoder_total)
for key in key_to_encoder:
    del arg_mlp[key]

for key in key_to_encoder:
    sp=key.split("encoder.")[1]
    arg_encoder[sp]=arg_encoder.pop(key)

for key in key_to_mlp:
    sp=key.split("mlp.")[1]
    arg_mlp[sp]=arg_mlp.pop(key)

print(arg_encoder)