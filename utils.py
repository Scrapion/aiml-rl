import numpy as np
import matplotlib.pyplot as plt

# Gestisce grafico dei rewards
def plot_rewards(stats, episode):

    x_axis = np.arange(1, episode+2, 1)

    figure, axis = plt.subplots(2,1)
    axis[0].plot(x_axis, np.zeros(x_axis.shape[0]), linestyle='dotted', color='purple', label='Zero Reference')
    axis[0].plot(x_axis, stats['max_rewards_home'], linestyle='dashdot', color='g', label='Max')
    axis[0].plot(x_axis, stats['mean_rewards_home'], color='g', label='Mean')
    axis[0].plot(x_axis, stats['min_rewards_home'], linestyle='dashed', color='g', label='Min')
    axis[0].set_title('Home Team')
    axis[0].legend()

    axis[1].plot(x_axis, np.zeros(x_axis.shape[0]), linestyle='dotted', color='purple', label='Zero Reference')    
    axis[1].plot(x_axis, stats['max_rewards_away'], linestyle='dashdot', color='r', label='Max')
    axis[1].plot(x_axis, stats['mean_rewards_away'], color='r', label='Mean')
    axis[1].plot(x_axis, stats['min_rewards_away'], linestyle='dashed', color='r', label='Min')
    axis[1].set_title('Away Team')
    axis[1].legend()

    plt.subplots_adjust(hspace = 0.3)
    plt.savefig('plots/plots.png')
    plt.close()

def plot_epsilon(stats, episode):

    epsilon_values = stats['epsilon']

    x_axis = np.arange(1, len(epsilon_values)+1)

    markers_on = [(e)*150 for e in range(1, episode+1) ]

    plt.plot(x_axis, epsilon_values, color='purple', marker='o', markevery=markers_on)
    plt.title('Epsilon value over learning iterations')
    plt.savefig('plots/epsilon.png')
    plt.close()