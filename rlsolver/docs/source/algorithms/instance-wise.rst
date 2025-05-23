.. _instance-wise:

逐实例方法（Instance-wise）
==============================

简介
----
Instance-wise 方法即 **依次对每个训练实例（或单条轨迹）进行采样和更新**，这是最传统的 RL 环境交互方式：

- 每次只运行一条环境实例  
- 完成一次 episode 后再开始下一条  
- 逻辑简单，CPU/GPU 混合场景友好  

实现细节
-----------
1. **环境封装**  
   - `RLSolver.environments.InstanceWiseEnv`  
   - 内部直接调用单环境 reset/step  
2. **步进流程**  
   - for _ in range(batch_size):  
       env.reset(); run one episode; 收集数据  
3. **数据融合**  
   - 将若干条轨迹串联后才输入到策略/价值网络  

使用示例
------------
.. code-block:: python

   from rlsolver.environments import InstanceWiseEnv
   env = InstanceWiseEnv(...)
   trajectories = []
   for _ in range(1024):
       obs = env.reset()
       done = False
       while not done:
           action = policy(obs)
           obs, reward, done, _ = env.step(action)
           # 存 trajectory
       trajectories.append(current_trajectory)
   # 将 trajectories 拼成 batch 进行网络训练

优缺点对比
-------------
- **优点**  
  - 实现简单，对显存和并行能力要求低  
  - 易于调试和单步监控  

- **缺点**  
  - CPU/GPU 切换频繁，吞吐量低  
  - 当 batch_size 很大时，整体耗时显著增加
