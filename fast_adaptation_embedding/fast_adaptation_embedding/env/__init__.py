
from gym.envs.registration import register

register(
	id='AntMuJoCoEnv_fastAdapt-v0',
	entry_point='fast_adaptation_embedding.env.ant_env:AntMuJoCoEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
)

register(
	id='Arm5dof-v0',
	entry_point='fast_adaptation_embedding.env.kinematic_arm:Arm_env',
	max_episode_steps=1000,
	reward_threshold=2500.0
)