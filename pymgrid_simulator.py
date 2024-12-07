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
    def __init__(self, learning_rate=0.1, discount_factor=0.9, randomization=0.3, grid_num=0, microgrid=None, end_step=None):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.randomization = randomization
        self.grid_num = grid_num
        self.input_microgrid=microgrid
        if microgrid != None:
            self.simulator = DiscreteMicrogridEnv.from_microgrid(microgrid)
        else:
            self.simulator = DiscreteMicrogridEnv.from_scenario(microgrid_number=grid_num)
        self.end_step = end_step
        self.theta = None
        # self.action_size = len(self.simulator.sample_action())
        self.action_space = range(len(self.simulator.actions_list))
        
        

    # def discritize_action_space(self):
    #     individual_actions = [x/10.0 for x in range(10)]
    #     discrete_actions = itertools.combinations_with_replacement(individual_actions, self.action_size)
    #     return [np.array(a) for a in discrete_actions]


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
        # energy_diff = 0
        # load_energy = 0
        # provided_energy = 0
        # co2_emissions = 0
        # if info.get("load"):
        #     load_energy = info["load"][0]["absorbed_energy"]
        #     features = np.append(features, load_energy)
        # if info.get("genset"):
        #     if info["genset"][0].get("provided_energy"):
        #         genset_provided_energy = info["genset"][0]["provided_energy"]
        #     else:
        #         genset_provided_energy = 0
        #     features = np.append(features, genset_provided_energy)
        #     provided_energy += genset_provided_energy
        #     genset_co2 = info["genset"][0]["co2_production"]
        #     features = np.append(features, genset_co2)
        #     co2_emissions += genset_co2
        # if info.get("battery"):
        #     if info["battery"][0].get("provided_energy"):
        #         battery_provided_energy = info["battery"][0]["provided_energy"]
        #     else:
        #         battery_provided_energy = 0
        #     features = np.append(features, battery_provided_energy)
        #     provided_energy += battery_provided_energy
        # if info.get("pv"):
        #     if info["pv"][0].get("provided_energy"):
        #         pv_provided_energy = info["pv"][0]["provided_energy"]
        #     else:
        #         pv_provided_energy = 0
        #     features = np.append(features, pv_provided_energy)
        #     provided_energy += pv_provided_energy
        #     pv_curtailment = info["pv"][0]["curtailment"]
        #     features = np.append(features, pv_curtailment)
        # if info.get("grid"):
        #     if info["grid"][0].get("provided_energy"):
        #         grid_provided_energy = info["grid"][0]["provided_energy"]
        #     else:
        #         grid_provided_energy = 0
        #     features = np.append(features, grid_provided_energy)
        #     provided_energy += grid_provided_energy
        #     grid_co2 = info["grid"][0]["co2_production"]
        #     features = np.append(features, grid_co2)
        #     co2_emissions += grid_co2
        # if info.get("unbalanced_energy"):
        #     if info["unbalanced_energy"][0].get("provided_energy"):
        #         unbalanced_provided_energy = info["unbalanced_energy"][0]["provided_energy"]
        #     else:
        #         unbalanced_provided_energy = 0
        #     features = np.append(features, battery_provided_energy)
        #     provided_energy += unbalanced_provided_energy
        # energy_diff = load_energy - provided_energy
        # features = np.append(features, energy_diff)
        # features = np.append(features, provided_energy)
        # features = np.append(features, co2_emissions)
        return features

    def q_value(self, state, action, info):
        """
        Compute Q-value for a given state and action using the parameterized model.
        """
        phi = self.feature_vector(state, action, info)
        # print("phi", phi.shape)
        if self.theta is None:
            self.theta = np.random.randn(phi.size)
        # print("theta", self.theta.shape)
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
            self.simulator.reset() # Reset the simulator for each episode
            total_error = 0
            grad_sum = 0
            self.randomization *= 0.8
            self.state_space = set()
            # take a random initial step
            action_index = np.random.randint(0, len(self.action_space))
            state, reward, done, info = self.simulator.step(self.action_space[action_index])

            while True:
                if np.random.random() > self.randomization:
                    q_array = [self.q_value(state, a, info) for a in self.action_space]
                    action_index = np.argmax(q_array)
                    action = self.action_space[action_index]
                    #print("policy-action", action)
                    #pdb.set_trace()
                else:
                    action_index = np.random.randint(0, len(self.action_space))
                    action = self.action_space[action_index]
                  # Define a reward function
                print("reward", reward)
                print("step", self.simulator.current_step)
                print("action", action)
                #pdb.set_trace()
                next_state, reward, done, next_info = self.simulator.step(action)  # Take an action with a fixed amount (e.g., 100 MW)
                # Get the next state
                if done or ((not (self.end_step == None)) and self.simulator.current_step >= self.end_step):
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

    def random_action_baseline(self):
        """
        Chooses random actions from the action space to use as a baseline.
        """
        total_reward = 0
        action_log = []
        timestamps = []

        if self.input_microgrid != None:
            test_simulator = DiscreteMicrogridEnv.from_microgrid(self.input_microgrid)
        else:
            test_simulator = DiscreteMicrogridEnv.from_scenario(microgrid_number=self.grid_num)

        #continuous
        # if self.input_microgrid != None:
        #     test_simulator = ContinuousMicrogridEnv.from_microgrid(self.input_microgrid)
        # else:
        #     test_simulator = ContinuousMicrogridEnv.from_scenario(microgrid_number=self.grid_num)
        # test_simulator = ContinuousMicrogridEnv.from_scenario(microgrid_number=self.grid_num)

        while True:
        # Select action based on the policy
            # Pick a random action
            action_index = np.random.randint(0, len(self.action_space))
            action = self.action_space[action_index]  
            
            # Perform the action in the simulator
            state, reward, done, info = test_simulator.step(action)
            total_reward += reward

            # Check for simulation end
            if done or ((not (self.end_step == None)) and test_simulator.current_step >= self.end_step):
                break

        microgrid_log = test_simulator.get_log()

        return total_reward, action_log, timestamps, microgrid_log
    
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

        if self.input_microgrid != None:
            test_simulator = DiscreteMicrogridEnv.from_microgrid(self.input_microgrid)
        else:
            test_simulator = DiscreteMicrogridEnv.from_scenario(microgrid_number=self.grid_num)

        # take a random action to start to be able to make an observation
        state, reward, done, info = test_simulator.step(test_simulator.sample_action())

        while True:

            # Compute Q-values for all possible actions
            q_values = [self.q_value(state, action, info) for action in self.action_space]


            # Select the action with the highest Q-value
            best_action = self.action_space[np.argmax(q_values)]
            print("step", test_simulator.current_step)
            # pdb.set_trace()
            action_log.append(test_simulator.convert_action(best_action))
            timestamps.append(time.time())

            # Perform the action in the simulator
            state, reward, done, info = test_simulator.step(best_action)
            total_reward += reward

            # Check for simulation end : 
            if done or ((not (self.end_step == None)) and test_simulator.current_step >= self.end_step):
                break

        return total_reward, action_log, timestamps, microgrid_log


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


class PymgridNeuralNetworkMDP:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, randomization=0.3, grid_num=0, microgrid=None, end_step=None):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.randomization = randomization
        self.grid_num = grid_num
        self.input_microgrid=microgrid
        self.end_step = end_step
        # discrete
        if microgrid != None:
            self.simulator = DiscreteMicrogridEnv.from_microgrid(microgrid)
        else:
            self.simulator = DiscreteMicrogridEnv.from_scenario(microgrid_number=grid_num)
        #continuous
        # if microgrid != None:
        #     self.simulator = ContinuousMicrogridEnv.from_microgrid(microgrid)
        # else:
        #     self.simulator = ContinuousMicrogridEnv.from_scenario(microgrid_number=grid_num)
        init_action = self.simulator.sample_action()
        #continuous
        # self.action_size = len(init_action)
        # self.action_space = self.discritize_action_space()
        #discrete
        self.action_space = range(len(self.simulator.actions_list))
        init_state, reward, done, init_info = self.simulator.step(init_action)
        self.init_state = init_state
        self.init_reward = reward
        self.init_info = init_info
        self.feature_size = len(self.feature_vector(init_state, init_action, init_info))
        # Initialize the Q-network and optimizer
        self.q_network = QNetwork(self.feature_size, 128, 1)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    # def discritize_action_space(self):
    #     individual_actions = [x/20.0 for x in range(20)]
    #     discrete_actions = itertools.combinations_with_replacement(individual_actions, self.action_size)
    #     return [np.array(a) for a in discrete_actions]


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
        # if info.get("load"):
        #     load_energy = info["load"][0]["absorbed_energy"]
        #     features = np.append(features, load_energy)
        # if info.get("genset"):
        #     if info["genset"][0].get("provided_energy"):
        #         genset_provided_energy = info["genset"][0]["provided_energy"]
        #     else:
        #         genset_provided_energy = 0
        #     features = np.append(features, genset_provided_energy)
        #     provided_energy += genset_provided_energy
        #     genset_co2 = info["genset"][0]["co2_production"]
        #     features = np.append(features, genset_co2)
        #     co2_emissions += genset_co2
        # if info.get("battery"):
        #     if info["battery"][0].get("provided_energy"):
        #         battery_provided_energy = info["battery"][0]["provided_energy"]
        #     else:
        #         battery_provided_energy = 0
        #     features = np.append(features, battery_provided_energy)
        #     provided_energy += battery_provided_energy
        # if info.get("pv"):
        #     if info["pv"][0].get("provided_energy"):
        #         pv_provided_energy = info["pv"][0]["provided_energy"]
        #     else:
        #         pv_provided_energy = 0
        #     features = np.append(features, pv_provided_energy)
        #     provided_energy += pv_provided_energy
        #     pv_curtailment = info["pv"][0]["curtailment"]
        #     features = np.append(features, pv_curtailment)
        # if info.get("grid"):
        #     if info["grid"][0].get("provided_energy"):
        #         grid_provided_energy = info["grid"][0]["provided_energy"]
        #     else:
        #         grid_provided_energy = 0
        #     features = np.append(features, grid_provided_energy)
        #     provided_energy += grid_provided_energy
        #     grid_co2 = info["grid"][0]["co2_production"]
        #     features = np.append(features, grid_co2)
        #     co2_emissions += grid_co2
        # if info.get("unbalanced_energy"):
        #     if info["unbalanced_energy"][0].get("provided_energy"):
        #         unbalanced_provided_energy = info["unbalanced_energy"][0]["provided_energy"]
        #     else:
        #         unbalanced_provided_energy = 0
        #     features = np.append(features, battery_provided_energy)
        #     provided_energy += unbalanced_provided_energy
        # energy_diff = load_energy - provided_energy
        # features = np.append(features, energy_diff)
        # features = np.append(features, provided_energy)
        # features = np.append(features, co2_emissions)
        return features

    def q_value(self, state, action, info):
        """
        Compute Q-value using the neural network.
        """
        state_action = self.feature_vector(state, action, info)
        state_action_tensor = torch.FloatTensor(state_action).unsqueeze(0)  # Batch dimension
        q_values = self.q_network(state_action_tensor)
        return q_values


    def update_parameters(self, state, action, info, reward, next_state, next_info):
        """
        Perform a Q-learning update using the neural network.
        """
        # Compute feature vector for the current state-action pair
        state_action = self.feature_vector(state, action, info)
        state_action_tensor = torch.FloatTensor(state_action).unsqueeze(0)  # Batch dimension

        # Compute the target Q-value
        next_q_values = []
        for a in self.action_space:
            next_state_action = self.feature_vector(next_state, a, next_info)
            next_q_values.append(self.q_value(next_state, a, next_info).item())
        target = float(reward + self.discount_factor * max(next_q_values))
    

        # Current Q-value prediction
        predicted_q = self.q_network(state_action_tensor)[0][0]  # Output for action index

        target_tensor = torch.tensor(target)

        # Loss and backpropagation
        loss = self.criterion(predicted_q, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
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

        if self.input_microgrid != None:
            test_simulator = DiscreteMicrogridEnv.from_microgrid(self.input_microgrid)
        else:
            test_simulator = DiscreteMicrogridEnv.from_scenario(microgrid_number=self.grid_num)
        
        # if self.input_microgrid != None:
        #     test_simulator = ContinuousMicrogridEnv.from_microgrid(self.input_microgrid)
        # else:
        #     test_simulator = ContinuousMicrogridEnv.from_scenario(microgrid_number=self.grid_num)

        # test_simulator = ContinuousMicrogridEnv.from_scenario(microgrid_number=self.grid_num)

        # take a random action to start to be able to make an observation
        state, reward, done, info = test_simulator.step(test_simulator.sample_action())

        while True:
        # Select action based on the policy
            q_values = []
            for action in self.action_space:
                state_action = self.feature_vector(state, action, info)
                state_action_tensor = torch.FloatTensor(state_action).unsqueeze(0)  # Batch dimension
                q_value = self.q_network(state_action_tensor).item()
                q_values.append(q_value)

            # Choose the action with the highest Q-value
            best_action = self.action_space[np.argmax(q_values)]
            action_log.append(best_action)
            timestamps.append(time.time())

             # Perform the action in the simulator
            state, reward, done, info = test_simulator.step(best_action)
            total_reward += reward

            # Check for simulation end
            if done or ((not (self.end_step == None)) and test_simulator.current_step >= self.end_step):
                break

        microgrid_log = test_simulator.get_log()

        return total_reward, action_log, timestamps, microgrid_log


    def random_action_baseline(self):
        """
        Chooses random actions from the action space to use as a baseline.
        """
        total_reward = 0
        action_log = []
        timestamps = []

        if self.input_microgrid != None:
            test_simulator = DiscreteMicrogridEnv.from_microgrid(self.input_microgrid)
        else:
            test_simulator = DiscreteMicrogridEnv.from_scenario(microgrid_number=self.grid_num)

        #continuous
        # if self.input_microgrid != None:
        #     test_simulator = ContinuousMicrogridEnv.from_microgrid(self.input_microgrid)
        # else:
        #     test_simulator = ContinuousMicrogridEnv.from_scenario(microgrid_number=self.grid_num)
        # test_simulator = ContinuousMicrogridEnv.from_scenario(microgrid_number=self.grid_num)

        while True:
        # Select action based on the policy
            # Pick a random action
            action_index = np.random.randint(0, len(self.action_space))
            action = self.action_space[action_index]  
            
            # Perform the action in the simulator
            state, reward, done, info = test_simulator.step(action)
            total_reward += reward

            # Check for simulation end
            if done or ((not (self.end_step == None)) and test_simulator.current_step >= self.end_step):
                break

        microgrid_log = test_simulator.get_log()

        return total_reward, action_log, timestamps, microgrid_log
   

    def train(self, episodes, epsilon=0.1):
        """
        Train the Q-network using the simulator.
        Parameters:
            - simulator (CampusEnergyReplaySimulator): The simulator instance.
            - episodes (int): Number of episodes to train.
            - epsilon (float): Epsilon-greedy policy parameter.
        Returns:
            - episode_rewards (list): List of total rewards for each episode.
        """
        episode_rewards = []

        for episode in tqdm(range(episodes)):
            self.simulator.reset() # Reset the simulator for each episode
            total_reward = 0

            state, reward, done, info = self.simulator.step(self.simulator.sample_action())

            while True:

                # Choose action using epsilon-greedy policy
                if np.random.random() < epsilon:
                    action_index = np.random.randint(0, len(self.action_space))
                    action = self.action_space[action_index]  # Explore
                else:
                    q_values = [self.q_value(state, action, info).item() for action in self.action_space]
                    action = self.action_space[np.argmax(q_values)]  # Exploit
                    # print("policy-action", action)

                next_state, reward, done, next_info = self.simulator.step(action)
                print("step", self.simulator.current_step)
                print("reward", reward)
                print("action", action, self.simulator.convert_action(action))
                total_reward += reward
                # Check for simulation end
                if done or ((not (self.end_step == None)) and self.simulator.current_step >= self.end_step):
                    break
            
                # Update Q-network
                loss = self.update_parameters(state, action, info, reward, next_state, next_info)
                state = next_state
                info = next_info
                

            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
            current_time = str(time.time())
            # torch.save(mdp.q_network.state_dict(), f"q_network{current_time}_episode{episode}.pth")
        return episode_rewards






# TO DO: clean up -- basic testing

if __name__ == "__main__":
    # learning_rate = 0.005
    # discount_factor = 0.85
    # num_episodes = 1

    # mdp = PymgridParameterizedMDP(learning_rate, discount_factor, grid_num=0, randomization=0.3)
    # theta, rewards = mdp.train(episodes=num_episodes)

    # print("Trained Parameters:", theta)
    # current_time = str(time.time())
    # saved_weights_file = f"weights{current_time}.npy"
    # np.save(saved_weights_file, theta)

    # total_reward, action_log, timestamps = mdp.run_inference()
    # print("Total Reward during Inference:", total_reward)
    # # print("Action Log:", action_log)
    # pd.DataFrame.from_dict({"DateTime": timestamps, "Action": action_log}).to_csv("action_log.csv")

    learning_rate = 0.05
    discount_factor = 0.6
    num_episodes = 1
    mdp = PymgridNeuralNetworkMDP(learning_rate, discount_factor, grid_num=19)
    rewards = mdp.train(episodes=num_episodes, epsilon=0.2)
    total_reward, action_log, timestamps, microgrid_log = mdp.run_inference()
    print("Total Reward during Inference:", total_reward)

    # from example
    microgrid_log[[('load', 0, 'load_met')]].droplevel(axis=1, level=1).plot()
    microgrid_log[[('net_load', 0, '')]].droplevel(axis=1, level=1).plot()

    # microgrid_log.to_csv("mg_log.csv")

    
    