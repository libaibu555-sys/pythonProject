# ...existing code...


# NumPy 示例（最轻量、纯数组）
def demo_numpy():
    import numpy as np
    a = np.array([1, 2, 3])
    b = np.zeros((2, 3))
    c = np.arange(6).reshape(2, 3)
    print("NumPy a:", a, "dtype:", a.dtype)
    print("NumPy b shape:", b.shape)
    print("NumPy c:\n", c)

# PyTorch 示例（如果安装了 torch）
def demo_torch():
    try:
        import torch
    except ImportError:
        print("torch 未安装，跳过 PyTorch 示例")
        return
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.zeros((2, 3), dtype=torch.float32)
    c = torch.arange(6).reshape(2, 3)
    print("Torch a:", a, "dtype:", a.dtype)
    print("Torch b shape:", b.shape)
    print("Torch c:\n", c)

# TensorFlow 示例（如果安装了 tensorflow）
def demo_tf():
    try:
        import tensorflow as tf
    except ImportError:
        print("tensorflow 未安装，跳过 TensorFlow 示例")
        return
    a = tf.constant([1, 2, 3])
    b = tf.zeros((2, 3), dtype=tf.float32)
    c = tf.reshape(tf.range(6), (2, 3))
    print("TF a:", a.numpy(), "dtype:", a.dtype)
    print("TF b shape:", b.shape)
    print("TF c:\n", c.numpy()) # 转为 NumPy 输出

if __name__ == "__main__":
    demo_numpy()
    demo_torch()
    demo_tf()