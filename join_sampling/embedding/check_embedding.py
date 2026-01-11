import pickle
import torch
import numpy as np

def quick_view(file_path):
    """快速查看 embedding 文件内容"""
    print(f"🔍 快速查看: {file_path}")
    print("=" * 60)
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"文件包含 {len(data)} 个 template")
    print()
    
    # 显示前5个 template
    for i, (key, emb) in enumerate(list(data.items())[:50]):
        print(f"Template {i+1}: {key}")
        
        # 转换 embedding
        if isinstance(emb, torch.Tensor):
            if emb.is_cuda:
                emb = emb.cpu()
            emb_np = emb.numpy()
        elif isinstance(emb, np.ndarray):
            emb_np = emb
        else:
            print(f"  未知 embedding 类型: {type(emb)}")
            continue
        
        print(f"  维度: {emb_np.shape}")
        print(f"  前5个值: {emb_np[:5].round(6)}")
        
        # 检查是否为零向量
        if np.allclose(emb_np, 0, atol=1e-8):
            print(f"  ⚠️  这是零向量!")
        else:
            # 显示一些统计信息
            print(f"  范数: {np.linalg.norm(emb_np):.4f}")
            print(f"  均值: {emb_np.mean():.6f}")
            print(f"  标准差: {emb_np.std():.6f}")
        
        print("-" * 40)

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    file_path = "/data1/xuyining/Sampler/join_sampling/embedding/results/small_skip7a/f2064923351388e03d492e7de432394e3da92046.pkl"
    quick_view(file_path)