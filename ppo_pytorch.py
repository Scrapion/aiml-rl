import numpy as np
import torch as T
from replay_buffer import PPOMemory
from networks import PPOActorNetwork, PPOCriticNetwork

class Agent:
    def __init__(self, input_dims, n_actions, discretization_grid, batch_size=64, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, n_epochs=10, fname='ppo_agent', import_model =False):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.fname = fname

        self.actor = PPOActorNetwork(n_actions, [input_dims], alpha)
        self.critic = PPOCriticNetwork([input_dims], alpha)
        self.memory = PPOMemory(batch_size, discretization_grid)

        if import_model:
            self.actor.load_state_dict('models/ppo/'+self.fname+'/actor')
            self.critic.load_state_dict('models/ppo/'+self.fname+'/critic')
       
    def remember(self, state, action, probs, vals, reward, done):
        print (action, 'action in remember')
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        T.save(self.actor.state_dict(), "models/ppo/"+self.fname+'/actor')
        T.save(self.critic.state_dict(), "models/ppo/"+self.fname+'/critic')

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)

        dist_move = self.actor(state)[0]
        dist_rotate = self.actor(state)[1]
        dist_chase = self.actor(state)[2]
        dist_cast = self.actor(state)[3]
        dist_change = self.actor(state)[4]

        value = self.critic(state)

        move = dist_move.sample()
        rotate = dist_rotate.sample()
        chase = dist_chase.sample()
        cast = dist_cast.sample()
        change = dist_change.sample()

        probs_move = T.squeeze(dist_move.log_prob(move)).item()
        move = T.squeeze(move).item()

        probs_rotate = T.squeeze(dist_rotate.log_prob(rotate)).item()
        rotate = T.squeeze(rotate).item()
        

        probs_chase = T.squeeze(dist_chase.log_prob(chase)).item()
        chase = T.squeeze(chase).item()
        

        probs_cast = T.squeeze(dist_cast.log_prob(cast)).item()
        cast = T.squeeze(cast).item()

        probs_change = T.squeeze(dist_change.log_prob(change)).item()
        change = T.squeeze(change).item()


        value = T.squeeze(value).item()

        probs = [probs_move, probs_rotate, probs_chase, probs_cast, probs_change]
        action = [move, rotate, chase, cast, change]

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)

                ##
                old_probs_move = T.tensor(old_prob_arr[0][batch]).to(self.actor.device)
                old_probs_rotate = T.tensor(old_prob_arr[1][batch]).to(self.actor.device)
                old_probs_chase = T.tensor(old_prob_arr[2][batch]).to(self.actor.device)
                old_probs_cast = T.tensor(old_prob_arr[3][batch]).to(self.actor.device)
                old_probs_change = T.tensor(old_prob_arr[4][batch]).to(self.actor.device)

                move = T.tensor(action_arr[0][batch]).to(self.actor.device)
                rotate = T.tensor(action_arr[1][batch]).to(self.actor.device)
                chase = T.tensor(action_arr[2][batch]).to(self.actor.device)
                cast = T.tensor(action_arr[3][batch]).to(self.actor.device)
                change = T.tensor(action_arr[4][batch]).to(self.actor.device)
                ##

                dist = self.actor(states)
                dist_move = dist[0]
                dist_rotate = dist[1]
                dist_chase = dist[2]
                dist_cast = dist[3]
                dist_change = dist[4]

                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs_move = dist_move.log_prob(move)
                new_probs_rotate = dist_rotate.log_prob(rotate)
                new_probs_chase = dist_chase.log_prob(chase)
                new_probs_cast = dist_cast.log_prob(cast)
                new_probs_change = dist_change.log_prob(change)

                move_ratio = new_probs_move.exp()/old_probs_move.exp()
                rotate_ratio = new_probs_rotate.exp()/old_probs_rotate.exp()
                chase_ratio = new_probs_chase.exp()/old_probs_chase.exp()
                cast_ratio = new_probs_cast.exp()/old_probs_cast.exp()
                change_ratio = new_probs_change.exp()/old_probs_change.exp()

                w_probs_move = advantage[batch] * move_ratio
                w_probs_rotate = advantage[batch] * rotate_ratio
                w_probs_chase = advantage[batch] * chase_ratio
                w_probs_cast = advantage[batch] * cast_ratio
                w_probs_change = advantage[batch] * change_ratio


                # Weighted Clipped
                wc_probs_move = T.clamp(move_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                wc_probs_rotate = T.clamp(rotate_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                wc_probs_chase = T.clamp(chase_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                wc_probs_cast = T.clamp(cast_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                wc_probs_change = T.clamp(change_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                
                move_loss = -T.min(w_probs_move, wc_probs_move).mean()
                rotate_loss = -T.min(w_probs_rotate, wc_probs_rotate).mean()
                chase_loss = -T.min(w_probs_chase, wc_probs_chase).mean()
                cast_loss = -T.min(w_probs_cast, wc_probs_cast).mean()
                change_loss = -T.min(w_probs_change, wc_probs_change).mean()

                actor_loss = move_loss + rotate_loss + chase_loss + cast_loss + change_loss

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()               


