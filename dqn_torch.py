import torch as T
import numpy as np
from copy import deepcopy

from networks import DeepQNetwork
from replay_buffer import ReplayBuffer




class Agent:
    def __init__(self, input_dims, n_actions, batch_size, action_space, discretization_grid, gamma=0.996, epsilon=1.0, lr=1e-4,   
                 max_mem_size=25000, replace = 100, eps_end=0.05, eps_dec=8e-5, fname='dqn_torch.h5', import_model = False):
        self.discretization_grid = discretization_grid
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.replace = replace
        self.lr = lr
        self.action_space = action_space
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100
        self.fname = fname

        self.memory = ReplayBuffer(max_mem_size, input_dims, n_actions, self.discretization_grid)
        self.Q_eval = DeepQNetwork(lr, action_size=n_actions, input_dims=[input_dims], fc1_dims=256, fc2_dims=256)
        self.Q_next = DeepQNetwork(lr, action_size=n_actions, input_dims=[input_dims], fc1_dims=256, fc2_dims=256)
        if import_model:
            self.Q_eval.load_state_dict('models/dqn/'+fname)
            self.Q_next.load_state_dict('models/dqn/'+fname)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation])).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            move = self.discretization_grid['MoveX'][np.argmax(actions[0].detach().cpu())]
            rotate = self.discretization_grid['Rotate'][np.argmax(actions[1].detach().cpu())]
            chase = self.discretization_grid['ChaseFocus'][np.argmax(actions[2].detach().cpu())]

            action = [move, rotate, chase, np.argmax(actions[3].detach().cpu()), np.argmax(actions[4].detach().cpu())]
        else:
            action = [np.random.choice(item) for item in self.discretization_grid.values()] + [np.random.randint(0,3), np.random.randint(0,7)]

        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        
        if self.memory.mem_cntr < self.batch_size:
            return

        self.replace_target_network()
        state, moves, rotates, chases, casts, changes, reward, new_state, terminal = self.memory.sample_buffer(self.batch_size)
        self.Q_eval.optimizer.zero_grad()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state = T.tensor(state).to(self.Q_eval.device)
        new_state = T.tensor(new_state).to(self.Q_eval.device)
        
        reward = T.tensor(reward).to(self.Q_eval.device)
        terminal = T.tensor(terminal).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state)
        (move_prd, rotate_prd, chase_prd, cast_prd, changes_prd) = q_eval

        q_move = move_prd[batch_index, moves]
        q_rotate = rotate_prd[batch_index, rotates]
        q_chase = chase_prd[batch_index, chases]
        q_cast = cast_prd[batch_index, casts]
        q_change = changes_prd[batch_index, changes]

        q_next = self.Q_next.forward(new_state)
        q_next[0][terminal] = 0.0
        q_next[1][terminal] = 0.0
        q_next[2][terminal] = 0.0
        q_next[3][terminal] = 0.0
        q_next[4][terminal] = 0.0

        q_target_move = reward + self.gamma*T.max(q_next[0], dim=1)[0]
        q_target_rotate = reward + self.gamma*T.max(q_next[1], dim=1)[0]
        q_target_chase = reward + self.gamma*T.max(q_next[2], dim=1)[0]
        q_target_cast = reward + self.gamma*T.max(q_next[3], dim=1)[0]
        q_target_change = reward + self.gamma*T.max(q_next[4], dim=1)[0]


        loss = (self.Q_eval.loss(q_target_move, q_move.double()) +\
                    self.Q_eval.loss(q_target_rotate, q_rotate.double())+\
                    self.Q_eval.loss(q_target_chase, q_chase.double()) +\
                    self.Q_eval.loss(q_target_cast, q_cast.double())+\
                    self.Q_eval.loss(q_target_change, q_change.double())).to(self.Q_eval.device)

        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
        
        return self.epsilon


    # Salva il modello
    def save_model(self):
        T.save(self.Q_eval.state_dict(), "models/dqn/"+self.fname)

    def replace_target_network(self):
        if self.iter_cntr != 0 and self.iter_cntr % self.replace == 0:
            self.Q_next.load_state_dict(deepcopy(self.Q_eval.state_dict()))
