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

                # ---- Vision QKV merged weight ----
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
                    hidden_size = tensor.shape[0] // 3
                    q_w = tensor[:hidden_size]
                    k_w = tensor[hidden_size:2 * hidden_size]
                    v_w = tensor[2 * hidden_size:]
                    if tensor.dtype != param.dtype:
                        q_w = q_w.to(param.dtype); k_w = k_w.to(param.dtype); v_w = v_w.to(param.dtype)
                    weight_loader(param, q_w, "q")
                    weight_loader(param, k_w, "k")
                    weight_loader(param, v_w, "v")
                    continue

                # ---- MoE fused expert weights (format B: [E,K,2N] / [E,N,K]) ----
                if ".experts.gate_up_proj" in target_name or ".experts.down_proj" in target_name:
                    tensor = f.get_tensor(weight_name)  # 3D
                    if ".experts.gate_up_proj" in target_name:
                        param_name = target_name.replace("experts.gate_up_proj", "experts.w13_weight")
                        t = tensor.transpose(-1, -2).contiguous()   # [E, 2N, K]
                        gate_w, up_w = t.chunk(2, dim=-2)           # each [E, N, K]
                        try:
                            param = model.get_parameter(param_name)
                        except (AttributeError, KeyError):
                            continue
                        if gate_w.dtype != param.dtype:
                            gate_w = gate_w.to(param.dtype); up_w = up_w.to(param.dtype)
                        E = tensor.shape[0]
                        for eid in range(E):
                            param.weight_loader(param, gate_w[eid], param_name, "w1", eid)
                            param.weight_loader(param, up_w[eid], param_name, "w3", eid)
                    else:
                        param_name = target_name.replace("experts.down_proj", "experts.w2_weight")
                        t = tensor.transpose(-1, -2).contiguous()   # [E, K, N]
                        try:
                            param = model.get_parameter(param_name)
                        except (AttributeError, KeyError):
                            continue
                        if t.dtype != param.dtype:
                            t = t.to(param.dtype)
                        E = tensor.shape[0]
                        for eid in range(E):
                            param.weight_loader(param, t[eid], param_name, "w2", eid)
                    continue

                # ---- packed modules (qkv_proj, gate_up_proj for dense layers) ----
                is_packed = False
                for k in packed_modules_mapping:
                    if k in weight_name:
                        if ".experts." in weight_name:
                            continue
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

                # ---- normal weights (incl. mlp.gate.weight router) ----
                try:
                    param = model.get_parameter(target_name)
                except (AttributeError, KeyError):
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                tensor = f.get_tensor(weight_name)
                if tensor.dtype != param.dtype:
                    tensor = tensor.to(param.dtype)
                weight_loader(param, tensor)



def load_qwen3_5_weights(model, path, config):
    import torch
    from nanovllm.utils.context import get_pstate
    tp = get_pstate().tp_size
    tp_rank = get_pstate().tp_rank

    text_cfg = config.hf_config.text_config if hasattr(config.hf_config, "text_config") else config.hf_config
    layer_types = list(text_cfg.layer_types)

    # GDN dims for qkv split
    num_k = text_cfg.linear_num_key_heads      # 16
    num_v = text_cfg.linear_num_value_heads    # 32
    hk = text_cfg.linear_key_head_dim          # 128
    hv = text_cfg.linear_value_head_dim        # 128
    key_dim = num_k * hk                         # 2048
    value_dim = num_v * hv                       # 4096

    tm = model.model
    layers = tm.layers

    def get_param(name):
        try:
            return model.get_parameter(name)
        except (AttributeError, KeyError):
            return None

    def shard_col(w):
        if tp == 1:
            return w
        n = w.shape[0]
        assert n % tp == 0, f"col shard {n} % {tp}"
        s = n // tp
        return w[tp_rank * s:(tp_rank + 1) * s]

    def shard_row(w):
        if tp == 1:
            return w
        n = w.shape[1]
        assert n % tp == 0, f"row shard {n} % {tp}"
        s = n // tp
        return w[:, tp_rank * s:(tp_rank + 1) * s]

    def shard_qkv(w):
        """w: [q(key_dim)+k(key_dim)+v(value_dim), ...]. Split each of q/k/v by tp, concat."""
        if tp == 1:
            return w
        q = w[:key_dim]
        k = w[key_dim:2 * key_dim]
        v = w[2 * key_dim:]
        kq = key_dim // tp
        kv = value_dim // tp
        q_s = q[tp_rank * kq:(tp_rank + 1) * kq]
        k_s = k[tp_rank * kq:(tp_rank + 1) * kq]
        v_s = v[tp_rank * kv:(tp_rank + 1) * kv]
        return torch.cat([q_s, k_s, v_s], dim=0)

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for wname in f.keys():
                if wname.startswith("mtp."):
                    continue

                if wname == "lm_head.weight":
                    p = model.lm_head.weight
                    t = f.get_tensor(wname)
                    (p.weight_loader(p, t.to(p.dtype)) if hasattr(p, "weight_loader")
                     else p.data.copy_(t.to(p.dtype)))
                    continue
                if wname == "model.language_model.embed_tokens.weight":
                    p = tm.embed_tokens.weight
                    t = f.get_tensor(wname)
                    (p.weight_loader(p, t.to(p.dtype)) if hasattr(p, "weight_loader")
                     else p.data.copy_(t.to(p.dtype)))
                    continue
                if wname == "model.language_model.norm.weight":
                    tm.norm.weight.data.copy_(f.get_tensor(wname).to(tm.norm.weight.dtype))
                    continue

                if wname.startswith("model.visual."):
                    vname = wname[len("model.visual."):]
                    if ".attn.qkv." in vname:
                        param = get_param("visual." + vname)
                        if param is None: continue
                        tensor = f.get_tensor(wname)
                        wl = getattr(param, "weight_loader", None)
                        if wl is None:
                            param.data.copy_(tensor.to(param.dtype)); continue
                        hs = tensor.shape[0] // 3
                        qw, kw, vw = tensor[:hs], tensor[hs:2*hs], tensor[2*hs:]
                        if tensor.dtype != param.dtype:
                            qw, kw, vw = qw.to(param.dtype), kw.to(param.dtype), vw.to(param.dtype)
                        wl(param, qw, "q"); wl(param, kw, "k"); wl(param, vw, "v")
                        continue
                    param = get_param("visual." + vname)
                    if param is None: continue
                    tensor = f.get_tensor(wname).to(param.dtype)
                    wl = getattr(param, "weight_loader", None)
                    if wl is not None: wl(param, tensor)
                    else: param.data.copy_(tensor)
                    continue

                if wname.startswith("model.language_model.layers."):
                    rest = wname[len("model.language_model.layers."):]
                    lidx = int(rest.split(".")[0])
                    sub = rest[len(str(lidx)) + 1:]
                    layer = layers[lidx]
                    tensor = f.get_tensor(wname)

                    if sub == "input_layernorm.weight":
                        layer.input_layernorm.weight.data.copy_(tensor.to(layer.input_layernorm.weight.dtype)); continue
                    if sub == "post_attention_layernorm.weight":
                        layer.post_attention_layernorm.weight.data.copy_(tensor.to(layer.post_attention_layernorm.weight.dtype)); continue

                    if sub == "mlp.gate_proj.weight":
                        layer.mlp.gate_proj.data.copy_(shard_col(tensor).to(layer.mlp.gate_proj.dtype)); continue
                    if sub == "mlp.up_proj.weight":
                        layer.mlp.up_proj.data.copy_(shard_col(tensor).to(layer.mlp.up_proj.dtype)); continue
                    if sub == "mlp.down_proj.weight":
                        layer.mlp.down_proj.data.copy_(shard_row(tensor).to(layer.mlp.down_proj.dtype)); continue

                    # ---- MoE MLP (Qwen3.5-MoE) ----
                    if sub == "mlp.gate.weight":
                        p = layer.mlp.gate.weight
                        p.data.copy_(tensor.to(p.dtype)); continue
                    if sub == "mlp.shared_expert_gate.weight":
                        p = layer.mlp.shared_expert_gate.weight
                        p.data.copy_(tensor.to(p.dtype)); continue
                    if sub == "mlp.shared_expert.gate_proj.weight":
                        layer.mlp.shared_expert.gate_proj.data.copy_(shard_col(tensor).to(layer.mlp.shared_expert.gate_proj.dtype)); continue
                    if sub == "mlp.shared_expert.up_proj.weight":
                        layer.mlp.shared_expert.up_proj.data.copy_(shard_col(tensor).to(layer.mlp.shared_expert.up_proj.dtype)); continue
                    if sub == "mlp.shared_expert.down_proj.weight":
                        layer.mlp.shared_expert.down_proj.data.copy_(shard_row(tensor).to(layer.mlp.shared_expert.down_proj.dtype)); continue
                    if sub == "mlp.experts.gate_up_proj":
                        # tensor: [E, 2*moe_inter, hidden] = [E, out, in]  (NO transpose)
                        exp = layer.mlp.experts
                        w13 = exp.w13_weight   # [E_local, 2*I, hidden]
                        E = tensor.shape[0]
                        I = exp.intermediate_size  # moe_inter (full, no TP for EP)
                        gate_w, up_w = tensor.chunk(2, dim=1)  # each [E, I, hidden]
                        gate_w = gate_w.to(w13.dtype); up_w = up_w.to(w13.dtype)
                        for eid in range(E):
                            exp.weight_loader(w13, gate_w[eid], "experts.w13_weight", "w1", eid)
                            exp.weight_loader(w13, up_w[eid], "experts.w13_weight", "w3", eid)
                        continue
                    if sub == "mlp.experts.down_proj":
                        # tensor: [E, hidden, moe_inter] = [E, out, in]  (NO transpose)
                        exp = layer.mlp.experts
                        w2 = exp.w2_weight   # [E_local, hidden, moe_inter]
                        E = tensor.shape[0]
                        t = tensor.to(w2.dtype)
                        for eid in range(E):
                            exp.weight_loader(w2, t[eid], "experts.w2_weight", "w2", eid)
                        continue
                        
                    if layer_types[lidx] == "linear_attention":
                        la = layer.linear_attn
                        if sub == "linear_attn.in_proj_qkv.weight":
                            la.in_proj_qkv.data.copy_(shard_qkv(tensor).to(la.in_proj_qkv.dtype)); continue  # 🚨 shard_qkv
                        if sub == "linear_attn.in_proj_z.weight":
                            la.in_proj_z.data.copy_(shard_col(tensor).to(la.in_proj_z.dtype)); continue
                        if sub == "linear_attn.in_proj_b.weight":
                            la.in_proj_b.data.copy_(shard_col(tensor).to(la.in_proj_b.dtype)); continue
                        if sub == "linear_attn.in_proj_a.weight":
                            la.in_proj_a.data.copy_(shard_col(tensor).to(la.in_proj_a.dtype)); continue
                        if sub == "linear_attn.conv1d.weight":
                            cw = tensor.squeeze(1)   # [conv_dim, kernel]
                            la.conv1d_weight.data.copy_(shard_qkv(cw).to(la.conv1d_weight.dtype)); continue  # 🚨 shard_qkv
                        if sub == "linear_attn.A_log":
                            la.A_log.data.copy_(shard_col(tensor).to(la.A_log.dtype)); continue
                        if sub == "linear_attn.dt_bias":
                            la.dt_bias.data.copy_(shard_col(tensor).to(la.dt_bias.dtype)); continue
                        if sub == "linear_attn.norm.weight":
                            la.norm.weight.data.copy_(tensor.to(la.norm.weight.dtype)); continue
                        if sub == "linear_attn.out_proj.weight":
                            la.out_proj.data.copy_(shard_row(tensor).to(la.out_proj.dtype)); continue
                    else:
                        at = layer.self_attn
                        if sub == "self_attn.q_proj.weight":
                            at.q_proj.data.copy_(shard_col(tensor).to(at.q_proj.dtype)); continue
                        if sub == "self_attn.k_proj.weight":
                            at.k_proj.data.copy_(shard_col(tensor).to(at.k_proj.dtype)); continue
                        if sub == "self_attn.v_proj.weight":
                            at.v_proj.data.copy_(shard_col(tensor).to(at.v_proj.dtype)); continue
                        if sub == "self_attn.o_proj.weight":
                            at.o_proj.data.copy_(shard_row(tensor).to(at.o_proj.dtype)); continue
                        if sub == "self_attn.q_norm.weight":
                            at.q_norm.weight.data.copy_(tensor.to(at.q_norm.weight.dtype)); continue
                        if sub == "self_attn.k_norm.weight":
                            at.k_norm.weight.data.copy_(tensor.to(at.k_norm.weight.dtype)); continue
                    continue