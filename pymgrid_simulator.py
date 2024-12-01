import pandas as pd
import numpy as np 
import time 
from tqdm import tqdm
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import pymgrid
from pymgrid.envs import DiscreteMicrogridEnv, ContinuousMicrogridEnv
import itertools




class PymgridParameterizedMDP:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, randomization=0.3, grid_num=0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.randomization = randomization
        self.grid_num = grid_num
        self.simulator = ContinuousMicrogridEnv.from_scenario(microgrid_number=grid_num)
        self.theta = None
        self.action_size = self.simulator.action_space.shape[0]
        self.action_space = self.discritize_action_space()

    def discritize_action_space(self):
        individual_actions = [x/10.0 for x in range(10)]
        discrete_actions = itertools.combinations_with_replacement(individual_actions, self.action_size)
        return [np.array(a) for a in discrete_actions]


    def feature_vector(self, state, action, info):
        """
        Compute feature vector for the given state and action.
        Example assumes state includes demand and generation.
        """
        grid_action = self.simulator.convert_action(action)
        features = state
        if grid_action.get("genset"):
            features = np.append(features, grid_action["genset"][0])
        if grid_action.get("grid"):
            features =np.append(features, grid_action["grid"][0])
        if grid_action.get("battery"):
            features = np.append(features, grid_action["battery"][0])
        energy_diff = 0
        load_energy = 0
        provided_energy = 0
        co2_emissions = 0
        if info.get("load"):
            load_energy = info["load"][0]["absorbed_energy"]
            features = np.append(features, load_energy)
        if info.get("genset"):
            if info["genset"][0].get("provided_energy"):
                genset_provided_energy = info["genset"][0]["provided_energy"]
            else:
                genset_provided_energy = 0
            features = np.append(features, genset_provided_energy)
            provided_energy += genset_provided_energy
            genset_co2 = info["genset"][0]["co2_production"]
            features = np.append(features, genset_co2)
            co2_emissions += genset_co2
        if info.get("battery"):
            if info["battery"][0].get("provided_energy"):
                battery_provided_energy = info["battery"][0]["provided_energy"]
            else:
                battery_provided_energy = 0
            features = np.append(features, battery_provided_energy)
            provided_energy += battery_provided_energy
        if info.get("pv"):
            if info["pv"][0].get("provided_energy"):
                pv_provided_energy = info["pv"][0]["provided_energy"]
            else:
                pv_provided_energy = 0
            features = np.append(features, pv_provided_energy)
            provided_energy += pv_provided_energy
            pv_curtailment = info["pv"][0]["curtailment"]
            features = np.append(features, pv_curtailment)
        if info.get("grid"):
            if info["grid"][0].get("provided_energy"):
                grid_provided_energy = info["grid"][0]["provided_energy"]
            else:
                grid_provided_energy = 0
            features = np.append(features, grid_provided_energy)
            provided_energy += grid_provided_energy
            grid_co2 = info["grid"][0]["co2_production"]
            features = np.append(features, grid_co2)
            co2_emissions += grid_co2
        if info.get("unbalanced_energy"):
            if info["unbalanced_energy"][0].get("provided_energy"):
                unbalanced_provided_energy = info["unbalanced_energy"][0]["provided_energy"]
            else:
                unbalanced_provided_energy = 0
            features = np.append(features, battery_provided_energy)
            provided_energy += unbalanced_provided_energy
        energy_diff = load_energy - provided_energy
        features = np.append(features, energy_diff)
        features = np.append(features, provided_energy)
        features = np.append(features, co2_emissions)
        return features

    def q_value(self, state, action, info):
        """
        Compute Q-value for a given state and action using the parameterized model.
        """
        phi = self.feature_vector(state, action, info)
        if self.theta is None:
            self.theta = np.random.randn(phi.size)
        return np.dot(self.theta, phi)

    def update_parameters(self, state, action, info, reward, next_state, next_info):
        """
        Perform a Q-learning parameter update.
        """
        phi = self.feature_vector(state, action, info)
        current_q = self.q_value(state, action, info)

        # Compute target Q-value
        next_q_values = [self.q_value(next_state, a, next_info) for a in self.action_space]
        target = reward + self.discount_factor * max(next_q_values)

        # TD error
        td_error = target - current_q

        # Calculate and clip the gradient
        gradient = td_error * phi
        max_gradient_norm = 1.0
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > max_gradient_norm:
            gradient *= max_gradient_norm / gradient_norm

        # Update parameters
        self.theta += self.learning_rate * gradient
        return td_error, gradient

    def train(self, episodes):
        """
        Train the parameterized MDP model using the simulator.
        """
        episode_rewards = []
        for episode in tqdm(range(episodes)):
            self.simulator =  ContinuousMicrogridEnv.from_scenario(microgrid_number=self.grid_num) # Reset the simulator for each episode
            total_error = 0
            grad_sum = 0
            self.randomization *= 0.8
            self.state_space = set()
            # take a random initial step
            curr_obs, reward, done, info = self.simulator.step(self.simulator.sample_action())

            while True:
                state = curr_obs

                if np.random.random() > self.randomization:
                    q_array = [self.q_value(state, a, info) for a in self.action_space]
                    action_index = np.argmax(q_array)
                    action = self.action_space[action_index]
                    #print("policy-action", action)
                    #pdb.set_trace()
                else:
                    action = self.simulator.sample_action()
                  # Define a reward function
                print("reward", reward)
                print("step", self.simulator.current_step)
                #pdb.set_trace()
                next_state, reward, done, next_info = self.simulator.step(action)  # Take an action with a fixed amount (e.g., 100 MW)
                # Get the next state
                if done:
                    break  # Stop if we've reached the end of the simulation

                # Update parameters
                td_error, grad = self.update_parameters(state, action, info, reward, next_state, next_info)
                state = next_state
                info = next_info
                grad_sum += np.sum(grad)
                total_error += td_error

            episode_rewards.append(total_error)
            print(f"Episode {episode + 1}: Total Reward = {total_error}, Total Gradient Sum = {grad_sum}")

        return self.theta, episode_rewards
    
    def run_inference(self):
        """
        Run inference using the parameterized MDP model.
        Parameters:
        Returns:
            - total_reward (float): Total reward accumulated during inference.
            - action_log (list): Log of actions taken during inference.
        """
        total_reward = 0
        action_log = []
        timestamps = []

        test_simulator = ContinuousMicrogridEnv.from_scenario(microgrid_number=self.grid_num)

        # take a random action to start to be able to make an observation
        state, reward, done, info = test_simulator.step(test_simulator.sample_action())

        while True:

            # Compute Q-values for all possible actions
            q_values = [self.q_value(state, action, info) for action in self.action_space]


            # Select the action with the highest Q-value
            best_action = self.action_space[np.argmax(q_values)]
            print("best_action", test_simulator.convert_action(best_action))
            # pdb.set_trace()
            action_log.append(test_simulator.convert_action(best_action))
            timestamps.append(time.time())

            # Perform the action in the simulator
            state, reward, done, info = test_simulator.step(best_action)
            total_reward += reward
            

            # Check for simulation end : temporarily setting to 100
            if done:
                break

        return total_reward, action_log, timestamps

# TO DO: UPDATE BELOW THIS LINE -----

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class NeuralNetworkMDP:
    def __init__(self, feature_size, action_space, learning_rate=0.001, discount_factor=0.9):
        self.feature_size = feature_size
        self.action_space = action_space
        self.discount_factor = discount_factor

        # Initialize the Q-network and optimizer
        self.q_network = QNetwork(feature_size, 128, len(action_space))
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def feature_vector(self, state, action):
        """
        Compute feature vector for the given state and action.
        """
        demand = state['demand']
        generation = state['generation']
        battery = state['battery_storage']
        return np.array([
            demand/50000,
            generation/50000,
            action/1750,
            battery/1750,
            state['time'].month/12,  # month
            state['time'].weekofyear/52,  # week
            state['time'].dayofweek/7,  # day of week
            (demand - generation)/50000,  # Interaction term
            (demand + action)/50000,  # Action-state interaction
            (generation + action)/50000,  # Action-state interaction
            (battery + action)/50000,  # Battery-action interaction
            (demand - generation + action)/50000  # Interaction term
        ])

    def q_value(self, state, action):
        """
        Compute Q-value using the neural network.
        """
        state_action = self.feature_vector(state, action)
        state_action_tensor = torch.FloatTensor(state_action).unsqueeze(0)  # Batch dimension
        q_values = self.q_network(state_action_tensor)
        return q_values

    def update_parameters(self, state, action, reward, next_state):
        """
        Perform a Q-learning update using the neural network.
        """
        # Compute feature vector for the current state-action pair
        state_action = self.feature_vector(state, action)
        state_action_tensor = torch.FloatTensor(state_action).unsqueeze(0)  # Batch dimension

        # Compute the target Q-value
        next_q_values = []
        for a in self.action_space:
            next_state_action = self.feature_vector(next_state, a)
            next_q_values.append(self.q_value(next_state, a).item())
        target = reward + self.discount_factor * max(next_q_values)

        # Current Q-value prediction
        predicted_q = self.q_network(state_action_tensor)[0][0]  # Output for action index

        # Loss and backpropagation
        loss = self.criterion(predicted_q, torch.tensor(target))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    @staticmethod
    def calculate_reward(state, action):
        """
        Example reward function based on minimizing unmet demand and storage overflow.
        """
        demand = state['demand']
        generation = state['generation']
        over_generation = max(0, generation - demand)
        battery_storage = state['battery_storage']
        if action >= 0:
            # Limit the storage amount to the available over-generation and battery capacity
            storable_energy = min(action, over_generation, 1750 - battery_storage)
            reward = -abs(demand - generation + storable_energy)*2.0  # Penalize unmet demand
            reward -= abs(action - storable_energy) # penalize over usage attempt of battery
        elif action < 0:
            # Limit the usage amount to available battery storage
            usable_energy = min(abs(action), battery_storage)
            reward = -abs(demand - generation - usable_energy)*2.0  # Penalize unmet demand
            reward -= abs(usable_energy + action) # penalize over usage attempt of battery
        
        reward -= max(0, 1750 - battery_storage)*0.125  # Penalize battery under-utlization

        return reward

    def run_inference(self, simulator):
        """
        Run inference using the trained policy.
        Parameters:
            - simulator (CampusEnergyReplaySimulator): The simulator instance.
        Returns:
            - total_reward (float): Total reward accumulated during inference.
            - action_log (list): Log of actions taken during inference.
        """
        total_reward = 0
        action_log = []
        timestamps = []

        while True:
            state = simulator.get_state()

            # Select action based on the policy
            q_values = []
            for action in self.action_space:
                state_action = self.feature_vector(state, action)
                state_action_tensor = torch.FloatTensor(state_action).unsqueeze(0)  # Batch dimension
                q_value = self.q_network(state_action_tensor).item()
                q_values.append(q_value)

            # Choose the action with the highest Q-value
            best_action = self.action_space[np.argmax(q_values)]
            action_log.append(best_action)
            timestamps.append(simulator.current_time)

            # Perform the action in the simulator
            reward = self.calculate_reward(state, best_action)
            total_reward += reward
            simulator.next_step(best_action)

            # Check for simulation end
            if simulator.current_time == simulator.data_source.index[0]:
                break

        return total_reward, action_log, timestamps




# TO DO: clean up -- basic testing

if __name__ == "__main__":
    learning_rate = 0.005
    discount_factor = 0.85
    num_episodes = 1

    mdp = PymgridParameterizedMDP(learning_rate, discount_factor, grid_num=1)
    theta, rewards = mdp.train(episodes=num_episodes)

    print("Trained Parameters:", theta)
    current_time = str(time.time())
    saved_weights_file = f"weights{current_time}.npy"
    np.save(saved_weights_file, theta)

    total_reward, action_log, timestamps = mdp.run_inference()
    print("Total Reward during Inference:", total_reward)
    # print("Action Log:", action_log)
    pd.DataFrame.from_dict({"DateTime": timestamps, "Action": action_log}).to_csv("action_log.csv")