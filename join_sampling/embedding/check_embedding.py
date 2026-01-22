# import os
# import pickle
# import torch
# import numpy as np
# from glob import glob


# def is_zero_embedding(emb, atol=1e-8):
#     """判断 embedding 是否为全零向量"""
#     if isinstance(emb, torch.Tensor):
#         if emb.is_cuda:
#             emb = emb.cpu()
#         emb = emb.numpy()
#     elif isinstance(emb, np.ndarray):
#         pass
#     else:
#         raise TypeError(f"Unsupported embedding type: {type(emb)}")

#     return np.allclose(emb, 0, atol=atol)


# def load_pkl(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)


# def compare_zero_embeddings(dir1, dir2):
#     """
#     遍历 dir1 中的 pkl 文件，
#     在 dir2 中找同名文件，对比 key 级别的零向量情况
#     """
#     pkl_files_1 = sorted(glob(os.path.join(dir1, "*.pkl")))

#     total_cnt = 0
#     zero_cnt_1 = 0
#     zero_cnt_2 = 0

#     missing_file_cnt = 0
#     missing_key_cnt = 0

#     for pkl1 in pkl_files_1:
#         fname = os.path.basename(pkl1)
#         pkl2 = os.path.join(dir2, fname)

#         if not os.path.exists(pkl2):
#             print(f"⚠️  dir2 中缺少文件: {fname}")
#             missing_file_cnt += 1
#             continue

#         data1 = load_pkl(pkl1)
#         data2 = load_pkl(pkl2)

#         for key, emb1 in data1.items():
#             if key not in data2:
#                 missing_key_cnt += 1
#                 continue

#             emb2 = data2[key]

#             total_cnt += 1

#             if is_zero_embedding(emb1):
#                 zero_cnt_1 += 1

#             if is_zero_embedding(emb2):
#                 zero_cnt_2 += 1

#     print("\n" + "=" * 60)
#     print("Zero Embedding Statistics")
#     print("=" * 60)
#     print(f"Total compared embeddings : {total_cnt}")
#     print()
#     print(f"Dir1 zero embeddings      : {zero_cnt_1}")
#     print(f"Dir1 zero ratio           : {zero_cnt_1 / total_cnt:.4%}")
#     print()
#     print(f"Dir2 zero embeddings      : {zero_cnt_2}")
#     print(f"Dir2 zero ratio           : {zero_cnt_2 / total_cnt:.4%}")
#     print()
#     print(f"Missing files in dir2     : {missing_file_cnt}")
#     print(f"Missing keys              : {missing_key_cnt}")
#     print("=" * 60)


# if __name__ == "__main__":
#     dir1 = "/data2/xuyining/Sampler/join_sampling/embedding/join/random_sample"
#     dir2 = "/data2/xuyining/Sampler/join_sampling/embedding/join/tree"

#     compare_zero_embeddings(dir1, dir2)


import os
import pickle
import torch
import numpy as np
from glob import glob


def is_zero_embedding(emb, atol=1e-8):
    """判断 embedding 是否为全零向量"""
    if isinstance(emb, torch.Tensor):
        if emb.is_cuda:
            emb = emb.cpu()
        emb = emb.numpy()
    elif isinstance(emb, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported embedding type: {type(emb)}")

    return np.allclose(emb, 0, atol=atol)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def check_zero_num(dir):
    """
    遍历 dir 中的 pkl 文件，
    统计零向量数量
    """
    pkl_files = sorted(glob(os.path.join(dir, "*.pkl")))

    total_cnt = 0
    zero_cnt = 0
    for pkl in pkl_files:
        data = load_pkl(pkl)
        for key, emb in data.items():
            if "n" in key or "t" in key or "mk" in key:
                continue
            total_cnt += 1
            if is_zero_embedding(emb):
                zero_cnt += 1

    print("\n" + "=" * 60)
    print("Zero Embedding Statistics")
    print("=" * 60)
    print(f"Total compared embeddings : {total_cnt}")
    print()
    print(f"Zero embeddings           : {zero_cnt}")
    print(f"Zero ratio                : {zero_cnt / total_cnt:.4%}")
    print("=" * 60)


if __name__ == "__main__":
    dir = "/data2/xuyining/Sampler/join_sampling/embedding/join/random_sample_no_mici"

    check_zero_num(dir)
