import numpy as np
from gym_derk.envs import DerkEnv
from gym_derk import TeamStatsKeys, DerkSession
from derkling_configs import get_config, get_config_fixed
from trainer_derk import Trainer
from utils import plot_rewards, plot_epsilon

load_models = False
ppo = False
n_arenas = 50
home_team, away_team = get_config(n_arenas)


env = DerkEnv(
  n_arenas=n_arenas,
  turbo_mode=True,
  home_team=home_team,
  away_team=away_team
)

stats = {
  'mean_rewards_home' : [],
  'min_rewards_home' : [],
  'max_rewards_home' : [],

  'mean_rewards_away' : [],
  'min_rewards_away' : [],
  'max_rewards_away' : [],

  'epsilon' : []
}

trainer = Trainer(action_space=env.action_space, observation_space=env.observation_space, load_models=load_models, ppo=ppo)

episodes = 100
for i in range(episodes):
  observation_n = env.reset()

  while True:
    
    action_n = trainer.act(observation_n)

    observation_n_new, reward_n, done_n, info_n = env.step(action_n)
    eps = trainer.teach(observation_n, action_n, reward_n, observation_n_new, done_n)
    stats['epsilon'].append(eps)

    observation_n = observation_n_new
    if all(done_n):
      break
  trainer.save_models()

  rewards_home_episode = [env.team_stats[t, TeamStatsKeys.Reward.value] for t in range(len(env.team_stats)) if t%2 == 0]
  stats['mean_rewards_home'].append(np.mean(rewards_home_episode))
  stats['min_rewards_home'].append(np.min(rewards_home_episode))
  stats['max_rewards_home'].append(np.max(rewards_home_episode))
  


  rewards_away_episode = [env.team_stats[t, TeamStatsKeys.Reward.value] for t in range(len(env.team_stats)) if t%2 == 1]
  stats['mean_rewards_away'].append(np.mean(rewards_away_episode))
  stats['min_rewards_away'].append(np.min(rewards_away_episode))
  stats['max_rewards_away'].append(np.max(rewards_away_episode))

  plot_rewards(stats, i)
  plot_epsilon(stats, i)


  print('Episode {}: Reward Home = {}, Reward Away = {}'.format(i, stats['mean_rewards_home'][i], stats['mean_rewards_away'][i]))

  


  

