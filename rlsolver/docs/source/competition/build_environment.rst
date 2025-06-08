Build MDP Model
===============

In RLSolver, we convert combinatorial optimization problems into Markov Decision Process (MDP) models, allowing reinforcement learning (RL) algorithms to solve them effectively.

The MDP model includes the following components:

- **State**: The current configuration of the graph or solution, e.g., spin vector in MaxCut.
- **Action**: The decision to flip a spin (0 ↔ 1) or take a special action such as "pass".
- **Reward**: The improvement in the objective function (e.g., increase in cut size).
- **Transition**: Applying the action updates the state to a new configuration.
- **Termination**: The episode ends when a maximum number of steps is reached or the solution stabilizes.

For example, in the MaxCut problem:

- A state is a binary vector representing partition assignment.
- An action is selecting a node and flipping its label.
- A reward is the change in cut value due to the flip.

Environment Setup
=================

This section explains how to define and customize your own environment in the ``rlsolver/methods/eco_s2v/src/envs/spinsystem.py`` file. The SpinSystem class supports both biased and unbiased environments and provides several extensible methods for configuration.

Key Functions
-------------

``reset(self, spins=None)``
    Initializes or resets the spin system for a new episode. This includes:
    
    - Generating a new graph from the graph generator.
    - Initializing the observable state matrix.
    - Sampling a valid spin configuration.
    - Setting up history and memory buffers (if enabled).
    
    This function is automatically called at the start of each episode.

``step(self, action)``
    Executes one step in the environment by flipping a spin or performing an extra action.
    
    Returns a tuple: ``(observation, reward, done, info)``. The step includes:

    - Updating the spin configuration.
    - Calculating reward based on score change or objective.
    - Updating observables and global features.
    - Checking if the episode is done (based on steps or irreversible spins).

``calculate_score(self, spins=None)``
    Computes the current objective value of the system.

    - If the optimization target is ``CUT``, it calculates the MaxCut value.
    - If the optimization target is ``ENERGY``, it returns the system energy.

Other Notable Components
------------------------

- ``action_space``: Defines the number of valid actions (spins + optional pass/randomize).
- ``observation_space``: Shape of the observable matrix used as the RL state.
- ``SpinSystemFactory``: Dynamically creates an instance of ``SpinSystemUnbiased`` or ``SpinSystemBiased``.

Customization Tips
------------------

To define your own environment:

1. Create a custom ``GraphGenerator`` class (see ``RandomGraphGenerator`` as reference).
2. Pass it to the ``SpinSystemFactory.get()`` method.
3. Adjust the parameters such as:

   - ``max_steps``: maximum steps per episode.
   - ``reward_signal``: DENSE, SINGLE, or CUSTOM_BLS.
   - ``spin_basis``: signed or binary spins.
   - ``extra_action``: include pass/randomize action if needed.
   - ``observables``: list of observables to monitor and update (e.g., reward available, distance from best score).
   - ``reversible_spins``: whether spins can be flipped more than once.

Example
-------

.. code-block:: python

   from rlsolver.methods.eco_s2v.src.envs.spinsystem import SpinSystemFactory
   from rlsolver.methods.eco_s2v.src.envs.util import RandomGraphGenerator

   env = SpinSystemFactory.get(
       graph_generator=RandomGraphGenerator(n_spins=20),
       max_steps=50,
       reward_signal=RewardSignal.DENSE,
       optimisation_target=OptimisationTarget.CUT,
       reversible_spins=True
   )

   obs = env.reset()
   obs, reward, done, _ = env.step(3)

The environment state (observation) is a stacked matrix including spin states, adjacency matrix, and (optionally) bias.

``SpinSystem`` serves as the core interface for training agents using structure2vec or other policy models.
