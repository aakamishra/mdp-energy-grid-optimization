import pandas as pd
import numpy as np 
import time 
from tqdm import tqdm
import pdb
import torch
import torch.nn as nn
import torch.optim as optim



DEFAULT_DATA_LOCATION = "ucsd-dataset/UCSD_Microgrid_Database/Data Files/DemandCharge.csv"


class CampusEnergyReplaySimulator:
    def __init__(self, data_location=DEFAULT_DATA_LOCATION, time_step=15, start_date="2018-01-01 00:00:00"):
        self.data_source = pd.read_csv(data_location, parse_dates=['DateTime'])
        self.data_source.set_index('DateTime', inplace=True)
        self.time_step = pd.Timedelta(minutes=time_step)
        self.current_time = pd.Timestamp(start_date)
        self.battery_storage = 0  # MW
        self.max_battery_storage = 1750  # MW
        self.stored_state = {}

        if self.current_time not in self.data_source.index:
            raise ValueError("Start date not found in the dataset.")

    def next_step(self, action):
        """
        Take a step in the simulation based on the provided action and amount.
        Actions:
            - "store": Save specified energy amount to the battery (up to available capacity).
            - "use": Use specified energy amount from the battery (up to available storage).
        Parameters:
            - action (str): "store" or "use"
            - amount (float): Amount of energy in MW to store or use
        """
        if self.current_time not in self.data_source.index:
            #print("ValueError('Current time is outside of the dataset range.' missing time index - ", self.current_time)
            generation =  self.stored_state["generation"]
            demand = self.stored_state["demand"]
            over_generation = self.stored_state["over_generation"]
        else:
            if type(self.data_source.loc[self.current_time].TotalCampusLoad) != np.float64:
                current_data = self.data_source.loc[self.current_time].iloc[0]
            else:
                current_data = self.data_source.loc[self.current_time]
            demand = current_data['TotalCampusLoad']
            generation = current_data['OnCampusGeneration']
            over_generation = max(0, generation - demand)

        if action >= 0:
            # Limit the storage amount to the available over-generation and battery capacity
            storable_energy = min(action, over_generation, self.max_battery_storage - self.battery_storage)
            self.battery_storage += storable_energy
        elif action < 0:
            # Limit the usage amount to available battery storage
            usable_energy = min(abs(action), self.battery_storage)
            self.battery_storage -= usable_energy
        else:
            raise ValueError("Invalid action. Choose either 'store' or 'use'.")

        # Advance time
        self.current_time += self.time_step

    def get_state(self):
        """
        Returns the current state of the system:
            - Current time
            - Demand (MW)
            - Generation (MW)
            - Over-generation (MW)
            - Battery storage (MW)
        """
        if self.current_time not in self.data_source.index:
            #print("ValueError('Current time is outside of the dataset range.' missing time index - ", self.current_time)
            #raise ValueError("Current time is outside of the dataset range.")
            return self.stored_state

        if type(self.data_source.loc[self.current_time].TotalCampusLoad) != np.float64:
            current_data = self.data_source.loc[self.current_time].iloc[0]
        else:
            current_data = self.data_source.loc[self.current_time]
        demand = current_data['TotalCampusLoad']
        generation = current_data['OnCampusGeneration']
        if type(current_data['OnCampusGeneration']) != np.float64:
            print(generation, self.current_time, type(current_data['OnCampusGeneration']))
        over_generation = max(0, generation - demand)
        #pdb.set_trace()
        self.stored_state = {
            "time": self.current_time,
            "demand": demand,
            "generation": generation,
            "over_generation": over_generation,
            "battery_storage": self.battery_storage
        }

        return self.stored_state



class ParameterizedMDP:
    def __init__(self, feature_size, learning_rate=0.1, discount_factor=0.9, randomization=0.3):
        self.theta = np.random.randn(feature_size)  # Initialize parameters randomly
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.randomization = randomization

    def feature_vector(self, state, action):
        """
        Compute feature vector for the given state and action.
        Example assumes state includes demand and generation.
        """
        demand = state['demand']
        generation = state['generation']
        battery = state['battery_storage']
        return np.array([
            demand,
            generation,
            action,
            battery,
            state['time'].month, # month
            state['time'].weekofyear, # week
            state['time'].dayofweek, # day of week
            demand - generation,  # Interaction term
            demand + action,  # Action-state interaction
            generation + action,  # Action-state interaction
            battery + action,  # Battery-action interaction
            demand - generation + action # Interaction term
        ])

    def q_value(self, state, action):
        """
        Compute Q-value for a given state and action using the parameterized model.
        """
        phi = self.feature_vector(state, action)
        return np.dot(self.theta, phi)

    def update_parameters(self, state, action, reward, next_state):
        """
        Perform a Q-learning parameter update.
        """
        phi = self.feature_vector(state, action)
        current_q = self.q_value(state, action)

        # Compute target Q-value
        possible_actions = np.arange(-1750, 1750, 25)
        next_q_values = [self.q_value(next_state, a) for a in possible_actions]
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

    def train(self, simulator, episodes, action_space):
        """
        Train the parameterized MDP model using the simulator.
        """
        episode_rewards = []
        for episode in tqdm(range(episodes)):
            simulator = CampusEnergyReplaySimulator()  # Reset the simulator for each episode
            total_error = 0
            grad_sum = 0
            self.randomization *= 0.8

            while True:
                state = simulator.get_state()

                if np.random.random() > self.randomization:
                    q_array = [self.q_value(state, a) for a in action_space]
                    action_index = np.argmax(q_array)
                    action = action_space[action_index]
                    #print("policy-action", action)
                    #pdb.set_trace()
                else:
                    action = np.random.choice(action_space)
                reward = self.calculate_reward(state, action)  # Define a reward function
                # print("reward", reward)
                #pdb.set_trace()
                simulator.next_step(action)  # Take an action with a fixed amount (e.g., 100 MW)
                # Get the next state
                if simulator.current_time == simulator.data_source.index[0]:
                    break  # Stop if we've reached the end of the simulation
                next_state = simulator.get_state()

                # Update parameters
                td_error, grad = self.update_parameters(state, action, reward, next_state)
                grad_sum += np.sum(grad)
                total_error += td_error

            episode_rewards.append(total_error)
            print(f"Episode {episode + 1}: Total Reward = {total_error}, Total Gradient Sum = {grad_sum}")

        return self.theta, episode_rewards
    
    def run_inference(self, simulator, action_space):
        """
        Run inference using the parameterized MDP model.
        Parameters:
            - simulator (CampusEnergyReplaySimulator): The simulator instance.
            - action_space (list or np.array): Possible actions to evaluate.
        Returns:
            - total_reward (float): Total reward accumulated during inference.
            - action_log (list): Log of actions taken during inference.
        """
        total_reward = 0
        action_log = []
        timestamps = []

        while True:
            state = simulator.get_state()

            # Compute Q-values for all possible actions
            q_values = [self.q_value(state, action) for action in action_space]


            # Select the action with the highest Q-value
            best_action = action_space[np.argmax(q_values)]
            pdb.set_trace()
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



if __name__ == "__main__":
    feature_size = 12  # Example feature size
    learning_rate = 0.005
    discount_factor = 0.85
    num_episodes = 30
    action_space = np.arange(-1750, 1750, 25)

    mdp = ParameterizedMDP(feature_size, learning_rate, discount_factor)
    simulator = CampusEnergyReplaySimulator()
    theta, rewards = mdp.train(simulator, episodes=num_episodes, action_space=action_space)

    print("Trained Parameters:", theta)
    current_time = str(time.time())
    saved_weights_file = f"weights{current_time}.npy"
    np.save(saved_weights_file, theta)

    # # Load trained parameters
    trained_parameters_path = saved_weights_file
    mdp = ParameterizedMDP(feature_size, learning_rate, discount_factor)
    mdp.theta = np.load(trained_parameters_path)

    simulator = CampusEnergyReplaySimulator()
    # Run inference
    total_reward, action_log, timestamps = mdp.run_inference(simulator, action_space)
    print("Total Reward during Inference:", total_reward)
    print("Action Log:", action_log)
    pd.DataFrame.from_dict({"DateTime": timestamps, "Action": action_log}).to_csv("action_log.csv")