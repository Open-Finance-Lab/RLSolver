.. _distribution-wise:

分布式采样方法（Distribution-wise）
===================================

简介
----
Distribution-wise 方法的核心思想是 **针对整个问题分布（distribution）一次性并行采样**，而不是逐个实例地运行。  
这种思路特别适合 GPU 上的大规模并行环境：

- **一次性生成 N 个样本**，利用 Tensor/CUDA 核心同时做环境步进  
- 跳过了循环“加载–运行–收集”开销  
- 提升了采样吞吐量（samples/sec）  

实现细节
-----------
1. **环境封装**  
   - `RLSolver.environments.DistributionWiseEnv`  
   - 接口和标准 Gym 环境一致，只是内部批量处理多条轨迹  
2. **并行机制**  
   - PyTorch/TensorFlow 张量并行运算  
   - 每个 step 同时推进所有轨迹  
3. **数据收集**  
   - 一次性返回形状为 `(batch_size, obs_dim)` 的观测张量  
   - 统一做探针、log 以及归一化  

使用示例
------------
.. code-block:: python

   from rlsolver.environments import DistributionWiseEnv
   env = DistributionWiseEnv(..., batch_size=1024)
   obs = env.reset()
   for _ in range(1000):
       actions = policy(obs)        # shape: [1024, action_dim]
       obs, rewards, dones, _ = env.step(actions)
       # 处理 dones，将对应轨迹 reset

优缺点对比
-------------
- **优点**  
  - 极低的单样本开销  
  - 内存占用可控（一次性预分配）

- **缺点**  
  - 需要足够大的 batch_size 才能发挥优势  
  - 不适合超大状态维度、显存受限场景
