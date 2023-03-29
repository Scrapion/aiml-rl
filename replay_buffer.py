import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discretization_grid):
        self.discretization_grid = discretization_grid
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype= np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype= np.float32)

        self.move_memory = np.zeros(self.mem_size, dtype=np.int8)
        self.rotate_memory = np.zeros(self.mem_size, dtype=np.int8)
        self.chase_memory = np.zeros(self.mem_size, dtype=np.int8)
        self.cast_memory = np.zeros(self.mem_size, dtype=np.int8)
        self.change_focus_memory = np.zeros(self.mem_size, dtype=np.int8)

        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)
    
    def store_transition(self, state, action, reward, state_, done):

        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_

        # store one hot encoding of actions, if appropriate
        self.move_memory[index] = np.where(self.discretization_grid['MoveX']==action[0])[0]

        self.rotate_memory[index] = np.where(self.discretization_grid['Rotate']==action[1])[0]

        self.chase_memory[index] = np.where(self.discretization_grid['ChaseFocus']==action[2])[0]

        self.cast_memory[index] = action[3]

        self.change_focus_memory[index] = action[4]


        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        moves = self.move_memory[batch]
        rotates = self.rotate_memory[batch]
        chases = self.chase_memory[batch]
        casts = self.cast_memory[batch]
        changes = self.change_focus_memory[batch]

        moves, rotates, chases, casts, changes

        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, moves, rotates, chases, casts, changes, rewards, states_, terminal


class PPOMemory:
    def __init__(self, batch_size, discretization_grid):
        self.states = []
        self.probs_move = []
        self.probs_rotate = []
        self.probs_chase = []
        self.probs_cast = []
        self.probs_change = []
        self.vals = []
        self.moves = []
        self.rotates = []
        self.chases = []
        self.casts = []
        self.changes = []
        self.rewards = []
        self.dones = []

        self.discretization_grid = discretization_grid

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                [np.array(self.moves),\
                np.array(self.rotates),\
                np.array(self.chases),\
                np.array(self.casts),\
                np.array(self.changes)],\
                [np.array(self.probs_move),\
                np.array(self.probs_rotate),\
                np.array(self.probs_chase),\
                np.array(self.probs_cast),\
                np.array(self.probs_change)],\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.moves.append(np.where(self.discretization_grid['MoveX']==action[0])[0]) 
        self.rotates.append(np.where(self.discretization_grid['Rotate']==action[1])[0])
        self.chases.append(np.where(self.discretization_grid['ChaseFocus']==action[2])[0])
        self.casts.append(action[3])
        self.changes.append(action[4])
        self.probs_move.append(probs[0])
        self.probs_rotate.append(probs[1])
        self.probs_chase.append(probs[2])
        self.probs_cast.append(probs[3])
        self.probs_change.append(probs[4])
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs_move = []
        self.probs_rotate = []
        self.probs_chase = []
        self.probs_cast = []
        self.probs_change = []
        self.moves = []
        self.rotates = []
        self.chases = []
        self.casts = []
        self.changes = []
        self.rewards = []
        self.dones = []
        self.vals = []