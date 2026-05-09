"""
Layer-by-layer precision comparison for Vision Encoder.
Compares HF Native (Single GPU) vs Our Implementation (TP=8).
Tracks error accumulation across all layers.
"""
import os
import io
import urllib.request
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from transformers import AutoProcessor, AutoConfig
from transformers import Qwen3VLForConditionalGeneration as HFModel

# 导入我们自己的模型加载器
from streamingvllm.models.qwen3_vl import load_qwen3_vl_model
from streamingvllm.config import Config

torch.set_default_dtype(torch.float32)

def download_image(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception as e:
        print(f"Failed to download image: {e}. Using dummy image.")
        return Image.new("RGB", (448, 448), color=(128, 128, 200))


def cmp(name, hf_tensor, our_tensor):
    """Compare two tensors and print detailed error distribution."""
    if hf_tensor.shape != our_tensor.shape:
        print(f"  [✗] {name:25s} | SHAPE MISMATCH: HF={hf_tensor.shape} vs Ours={our_tensor.shape}")
        return False
        
    diff = (hf_tensor.float() - our_tensor.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    total_elements = diff.numel()
    
    # 计算不同阈值下的误差占比
    pct_gt_01 = (diff > 0.1).sum().item() / total_elements * 100
    pct_gt_05 = (diff > 0.5).sum().item() / total_elements * 100
    pct_gt_10 = (diff > 1.0).sum().item() / total_elements * 100
    
    if max_diff < 0.01:
        status = "✅ MATCH"
    elif max_diff < 0.1:
        status = "⚠️ CLOSE"
    elif pct_gt_10 > 1.0: # 如果超过 1% 的元素误差大于 1.0，才认为是真正的严重错误
        status = "❌ LARGE"
    else:
        status = "⚠️ ACCUM"  # 只是极个别元素的误差累积
        
    print(f"  {status} | {name:25s} | max: {max_diff:.4f} | mean: {mean_diff:.4f} | >0.1: {pct_gt_01:.4f}% | >0.5: {pct_gt_05:.4f}% | >1.0: {pct_gt_10:.4f}%")
    
    return max_diff


def worker(rank, world_size, model_path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    processor = AutoProcessor.from_pretrained(model_path)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = download_image(url)
    
    inputs = processor(
        text=["<|vision_start|><|image_pad|><|vision_end|>"], 
        images=[img], return_tensors="pt", padding=True
    )
    pv = inputs["pixel_values"].to(torch.bfloat16).cuda()
    thw = inputs["image_grid_thw"].cuda()
    thw_list = thw.tolist()

    if rank == 0:
        print("\n" + "="*80)
        print("Loading HuggingFace Native Model (Rank 0)...")
        hf_model = HFModel.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
        hf_vis = hf_model.model.visual
        hf_vis.eval()

    dist.barrier()

    if rank == 0:
        print(f"Loading Our Model with TP={world_size}...")
        
    config = Config.__new__(Config)
    config.model = model_path
    config.hf_config = AutoConfig.from_pretrained(model_path)
    config.is_multimodal = True
    config.tensor_parallel_size = world_size
    
    our_model = load_qwen3_vl_model(model_path, config)
    our_vis = our_model.visual
    our_vis.eval()
    vc = config.hf_config.vision_config

    dist.barrier()
    if rank == 0:
        print("\n" + "="*80)
        print(f"Error Accumulation Tracking (TP={world_size})")
        print("="*80)

    with torch.no_grad():
        # --- 1. Initial Processing ---
        our_pe = our_vis.patch_embed(pv)
        our_pos = our_vis.fast_pos_embed_interpolate(thw_list)
        our_h = our_pe.reshape(-1, vc.hidden_size) + our_pos
        our_cos, our_sin = our_vis.rot_pos_emb(thw_list)
        
        ga = np.array(thw_list, dtype=np.int32)
        seq_per = ga[:, 1] * ga[:, 2]
        cu = np.concatenate([np.zeros(1, dtype=np.int32), np.repeat(seq_per, ga[:, 0]).cumsum().astype(np.int32)])
        cu_sl = torch.from_numpy(cu).cuda()
        max_sl = int((cu_sl[1:] - cu_sl[:-1]).max().item())

        if rank == 0:
            hf_pe = hf_vis.patch_embed(pv)
            hf_pos = hf_vis.fast_pos_embed_interpolate(thw)
            hf_h = hf_pe.reshape(-1, vc.hidden_size) + hf_pos
            hf_rope = hf_vis.rot_pos_emb(thw)
            hf_doubled = torch.cat([hf_rope, hf_rope], dim=-1)
            hf_cos, hf_sin = hf_doubled.cos(), hf_doubled.sin()

            cmp("Initial Hidden + Pos", hf_h, our_h)

        # --- 2. Transformer Blocks ---
        our_x = our_h.clone()
        if rank == 0:
            hf_x = hf_h.clone()

        hf_ds_features = []
        our_ds_features = []

        # 记录最大误差趋势
        max_diffs = []

        for i in range(vc.depth):
            our_x_in = our_x.clone()
            
            # Our Block Forward
            our_x = our_vis.blocks[i](our_x, cu_sl, our_cos, our_sin, max_sl)
            
            if rank == 0:
                hf_x_in = hf_x.clone()  # 🚨 确保在这里定义 🚨
                
                # HF Block Forward
                hf_x = hf_vis.blocks[i](hf_x, cu_seqlens=cu_sl, position_embeddings=(hf_cos, hf_sin))
                
                # Check Block Output
                hf_cmp = hf_x.squeeze(1) if hf_x.dim() == 3 else hf_x
                diff = cmp(f"Block {i:02d} Final", hf_cmp, our_x)
                max_diffs.append(diff)

                # 🚨 如果在第 9 层发现巨大误差，立刻解剖内部！🚨
                if i == 26 and diff > 1.0:
                    print(f"\n{'='*80}")
                    print(f"ANATOMY OF BLOCK 26")
                    print(f"{'='*80}")
                    
                    hf_b = hf_vis.blocks[i]
                    our_b = our_vis.blocks[i]
                    
                    # 1. Input
                    hf_in_cmp = hf_x_in.squeeze(1) if hf_x_in.dim() == 3 else hf_x_in
                    cmp("Input to Block 09", hf_in_cmp, our_x_in)
                    
                    # 2. Norm1
                    hf_n1 = hf_b.norm1(hf_x_in)
                    our_n1 = our_b.norm1(our_x_in)
                    hf_n1_cmp = hf_n1.squeeze(1) if hf_n1.dim() == 3 else hf_n1
                    cmp("Norm1", hf_n1_cmp, our_n1)
                    
                    # 3. Attention
                    hf_attn_out = hf_b.attn(
                        hf_n1, cu_seqlens=cu_sl, position_embeddings=(hf_cos, hf_sin)
                    )
                    our_attn_out = our_b.attn(our_n1, cu_sl, our_cos, our_sin, max_sl)
                    hf_attn_cmp = hf_attn_out.squeeze(1) if hf_attn_out.dim() == 3 else hf_attn_out
                    cmp("Attention Out", hf_attn_cmp, our_attn_out)
                    
                    # 4. Residual 1 + Norm2
                    hf_h_after_attn = hf_x_in + hf_attn_out
                    our_h_after_attn = our_x_in + our_attn_out
                    
                    hf_n2 = hf_b.norm2(hf_h_after_attn)
                    our_n2 = our_b.norm2(our_h_after_attn)
                    hf_n2_cmp = hf_n2.squeeze(1) if hf_n2.dim() == 3 else hf_n2
                    cmp("Norm2", hf_n2_cmp, our_n2)
                    
                    # 5. MLP
                    hf_mlp_out = hf_b.mlp(hf_n2)
                    our_mlp_out = our_b.mlp(our_n2)
                    hf_mlp_cmp = hf_mlp_out.squeeze(1) if hf_mlp_out.dim() == 3 else hf_mlp_out
                    cmp("MLP Out", hf_mlp_cmp, our_mlp_out)
                    
                    # Stop here to analyze
                    # break

            # DeepStack Check (保持不变)
            if i in vc.deepstack_visual_indexes:
                ds_idx = vc.deepstack_visual_indexes.index(i)
                our_ds = our_vis.deepstack_merger_list[ds_idx](our_x.clone())
                our_ds_features.append(our_ds)
                
                if rank == 0:
                    hf_ds = hf_vis.deepstack_merger_list[ds_idx](hf_x.clone())
                    hf_ds_cmp = hf_ds.squeeze(1) if hf_ds.dim() == 3 else hf_ds
                    cmp(f"  -> DeepStack L{ds_idx}", hf_ds_cmp, our_ds)

        # --- 3. Final Merger ---
        our_merged = our_vis.merger(our_x)
        if rank == 0:
            hf_merged = hf_vis.merger(hf_x)
            cmp("\nFinal Merger (Main)", hf_merged, our_merged)

            print("\n" + "="*80)
            print("Error Accumulation Summary:")
            print("="*80)
            for i, d in enumerate(max_diffs):
                print(f"Block {i:02d}: max_diff = {d:.6f}")

    dist.barrier()
    if rank == 0:
        print("\nDone.")

if __name__ == "__main__":
    # 请确认模型路径
    MODEL_PATH = "./Qwen/Qwen3-VL-8B-Instruct"
    
    # 🚨 强制设为 8 卡 TP 测试 🚨
    WORLD_SIZE = 8
    
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(WORLD_SIZE, MODEL_PATH), nprocs=WORLD_SIZE, join=True)
