import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, path: str, name_mapping=None):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                target_name = weight_name
                if name_mapping is not None:
                    target_name = name_mapping(target_name)
                    if target_name is None:
                        continue
                
                # 🚨 核心修复：特殊处理 Vision Encoder 的 QKV 合并权重 🚨
                if "visual." in target_name and ".attn.qkv." in target_name:
                    tensor = f.get_tensor(weight_name)
                    
                    try:
                        param = model.get_parameter(target_name)
                    except (AttributeError, KeyError):
                        continue
                        
                    weight_loader = getattr(param, "weight_loader", None)
                    if weight_loader is None:
                        if tensor.dtype != param.dtype:
                            tensor = tensor.to(param.dtype)
                        param.data.copy_(tensor)
                        continue
                    
                    # 开始切分 [3 * hidden_size, ...]
                    hidden_size = tensor.shape[0] // 3
                    q_weight = tensor[:hidden_size]
                    k_weight = tensor[hidden_size:2*hidden_size]
                    v_weight = tensor[2*hidden_size:]
                    
                    if tensor.dtype != param.dtype:
                        q_weight = q_weight.to(param.dtype)
                        k_weight = k_weight.to(param.dtype)
                        v_weight = v_weight.to(param.dtype)
                    
                    # 依次调用 QKVParallelLinear 的 weight_loader
                    weight_loader(param, q_weight, "q")
                    weight_loader(param, k_weight, "k")
                    weight_loader(param, v_weight, "v")
                    
                    # 🚨 处理完就 continue，跳过下面的逻辑 🚨
                    continue

                # 处理语言模型的 packed modules (如 qkv_proj, gate_up_proj)
                is_packed = False
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = target_name.replace(k, v) if k in target_name else target_name
                        try:
                            param = model.get_parameter(param_name)
                        except (AttributeError, KeyError):
                            continue
                        weight_loader = getattr(param, "weight_loader")
                        tensor = f.get_tensor(weight_name)
                        if tensor.dtype != param.dtype:
                            tensor = tensor.to(param.dtype)
                        weight_loader(param, tensor, shard_id)
                        is_packed = True
                        break
                
                if is_packed:
                    continue
                
                # 普通权重的加载
                try:
                    param = model.get_parameter(target_name)
                except (AttributeError, KeyError):
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                tensor = f.get_tensor(weight_name)
                if tensor.dtype != param.dtype:
                    tensor = tensor.to(param.dtype)
                weight_loader(param, tensor)
