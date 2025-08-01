# Max-Cut PPO 训练与评估

## 训练

单GPU训练：
```bash
python train_ddp.py
```

多GPU训练：
```bash
python launch.py
```

训练完成后生成 `model.pth`

## 参数配置

修改 `config.py` 中的参数：

```python
# 主要参数
epochs = 1000          # 训练轮数
batch_size = 8192      # 批次大小
lr = 2e-4             # 学习率
num_parallel_envs = 8  # 并行环境数

# 环境参数
episode_length_multiplier = 2  # 最大步数倍数
tabu_tenure = 10              # Tabu长度
```

## 评估

将 `evaluate.py` 放置在 `rlsolver/methods/maxcut/` 目录下：

```bash
cd rlsolver/methods/maxcut/
python evaluate.py
```

评估脚本会：
- 加载 `model.pth`
- 读取 `../../data` 下的测试图
- 保存结果到 `../../result`