import numpy as np
from numba import njit
from gym_derk import ActionKeys
from gym.spaces import Box, Discrete

from dqn_torch import Agent as DQN_Agent
from ppo_pytorch import Agent as PPO_Agent

class Trainer(object):
    def __init__(self, action_space, observation_space, load_models=False, ppo=False):
        self.action_space = action_space
        self.observation_space = observation_space
        self.step = 0
        self.learning_iteration = 0
        self.ppo = ppo
        self.N = 20


        #Crea uno spazio discreto per la discretizzazione delle azioni continue.
        #Le azioni continue vengno discretizzate in 10 punti.
        
        self.discretization_grid = {
            action.name: np.linspace(
                action_space[action.value].low,
                action_space[action.value].high,
                10,)

        for action in (ActionKeys.MoveX, ActionKeys.Rotate, ActionKeys.ChaseFocus)
        }


        # Costruisce vettore action_size per mantenere in memoria le dimensioni degli action_space.
        action_size = []
        for space, action in zip(action_space, ActionKeys):
            if action is ActionKeys.MoveX or action is ActionKeys.Rotate or action is ActionKeys.ChaseFocus:
                size = 10
            else:
                size = space.n
            action_size.append(size)

        # Calcolo grandezza observation space (Gioco con le strutture dati Box)
        if isinstance(observation_space, Box):
            (self.observation_size,) = observation_space.shape
        else:
            self.observation_size = observation_space


        # Istanzio gli agenti
        roles = ['DPS', 'Healer', 'Tank']
        if self.ppo:
            self.agents = [PPO_Agent(self.observation_size, action_size, self.discretization_grid, batch_size=64, fname=role, import_model=load_models) for role in roles]
        else:
            self.agents = [DQN_Agent(self.observation_size, action_size, 1000, action_space, self.discretization_grid, fname=role, import_model=load_models) for role in roles]


    # Chiedo agli agenti di scegliere un'azione
    def act(self, observation_n):
        actions = []
        for index, observation in enumerate(observation_n):
            actions.append(self.agents[index % 3].choose_action(observation))
        return actions

    # Chiedo agli agenti di scegliere un'azione
    def act_ppo(self, observation_n):
        actions, probs, vals = [], [], []
        for index, observation in enumerate(observation_n):
            action, prob, val = self.agents[index % 3].choose_action(observation)
            actions.append(action)
            probs.append(prob)
            vals.append(val)
        return actions, probs, vals

    # Prima riempie le memorie di ogni agente, poi lancia il learning. Due cicli anzichè uno, per favorire il multi-arena.
    def teach(self, observation_n, action, reward_n, observation_n_new, done):
        
        for index in range(len(observation_n)):
            self.agents[index % 3].remember(observation_n[index], action[index], reward_n[index], observation_n_new[index], done[0]) # Done è un figlio della merda

        for agent in self.agents:
            epsilon = agent.learn()

        return epsilon
    
    # Versione di teach per PPO 
    def teach(self, observation_n, action, probs_n, val_n, reward_n, done):
        self.step += 1
        for index in range(len(observation_n)):
            self.agents[index % 3].remember(observation_n[index], action[index], probs_n[index], val_n[index], reward_n[index], done[0]) # Done è un figlio della merda
        if self.step % self.N == 0:
            self.learning_iteration += 1
            for agent in self.agents:
                agent.learn()
       
        return self.learning_iteration

    # Salva i modelli su file per futuro caricamento.
    def save_models(self):
        for agent in self.agents:
            agent.save_model()

    # Riscritte in modo più pythonic
    """
    def teach(self, observation_n, action, reward_n, observation_n_new, done):
        for index in range(len(observation_n)):
            if (index % 3 == 0):
                self.agent_1.remember(observation_n[index], action[index], reward_n[index], observation_n_new[index], done[index])
            elif (index % 3 == 1):
                self.agent_2.remember(observation_n[index], action[index], reward_n[index], observation_n_new[index], done[index])
            else:
                self.agent_3.remember(observation_n[index], action[index], reward_n[index], observation_n_new[index], done[index])

        self.agent_1.learn()
        self.agent_2.learn()
        self.agent_3.learn()
    """

    """
    def act(self, observation_n):
        actions = []
        for index, observation in enumerate(observation_n):
            if (index % 3 == 0):
                actions.append(self.agent_1.choose_action(observation))
            elif (index % 3 == 1):
                actions.append(self.agent_2.choose_action(observation))
            else:
                actions.append(self.agent_3.choose_action(observation))
        return actions
    """