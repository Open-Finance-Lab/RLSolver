运行脚本说明
============

本节说明如何使用 `run.py` 脚本启动训练任务。

命令行参数：
------------

- `--env`: 指定使用的环境（如 TSP、MaxCut）
- `--agent`: 指定使用的 RL 算法（如 DQN、PPO）
- `--episodes`: 训练轮数

示例：
------

.. code-block:: bash

    python run.py --env TSP --agent DQN --episodes 500
