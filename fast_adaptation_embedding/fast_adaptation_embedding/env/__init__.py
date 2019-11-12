
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

register(
	id='MinitaurBulletEnv_fastAdapt-v0',
	entry_point='fast_adaptation_embedding.env.minitaur_env:MinitaurBulletEnv',
	max_episode_steps=1000,
	reward_threshold=5.0
)

register(
	id='MinitaurGymEnv_fastAdapt-v0',
	entry_point='fast_adaptation_embedding.env.minitaur_gym_env:MinitaurGymEnv',
	max_episode_steps=1000,
	reward_threshold=5.0
)

register(
	id='PexodAnt-v0',
	entry_point='fast_adaptation_embedding.env.pexod_ant:PexodAnt_env',
	max_episode_steps=10000,
	reward_threshold=10000.0
)

register(
	id='PexodAnt-v2',
	entry_point='fast_adaptation_embedding.env.pexod_ant_v2:PexodAnt_env',
	max_episode_steps=10000,
	reward_threshold=10000.0
)

register(
	id='PexodQuad-v0',
	entry_point='fast_adaptation_embedding.env.pexod_quad_env:PexodQuad_env',
	max_episode_steps=10000,
	reward_threshold=10000.0
)
