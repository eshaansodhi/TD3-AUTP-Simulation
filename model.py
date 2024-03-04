import numpy as np
import matplotlib.pyplot as plt
import random
import os
from gym import spaces
import tensorflow as tf
from config import (
    PROPAGATION_D, PROPAGATION_C, PATH_LOSS_ALPHA, CARRIER_FREQUENCY, NOISE_POWER, HEIGHT,
    MAX_BANDWIDTH, GAMMA_FUTURE_REWARDS, LOS_N1, NLOS_N2, ACTOR_ALPHA, CRITIC_ALPHA,
    DECAY_ACTION_RANDOMNESS, GAMMA, SELF_REPLACEMENT_VALUE, M, X_MAX, Y_MAX,
    SCHEDULE_MAX, MAX_SPEED, LAST_TIME_IOTD, MAX_DATA
)


x_max = 50
y_max = 50
num_iotd = 10

IOTD_LOC = [[random.uniform(0, x_max), random.uniform(0, y_max)] for _ in range(num_iotd)]

from keras.optimizers.legacy import Adam

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.idx = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.idx] = (state, action, reward, next_state, done)
            self.idx = (self.idx + 1) % self.buffer_size

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None  # Return None if buffer doesn't have enough samples
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for experience in batch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)



class TD3Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim  # Added to store action_dim

        self.actor = self.build_actor_network(state_dim, action_dim)
        self.critic1 = self.build_critic_network(state_dim, action_dim)
        self.critic2 = self.build_critic_network(state_dim, action_dim)
        self.target_actor = self.build_actor_network(state_dim, action_dim)
        self.target_critic1 = self.build_critic_network(state_dim, action_dim)
        self.target_critic2 = self.build_critic_network(state_dim, action_dim)
        
        self.actor_optimizer = Adam(ACTOR_ALPHA)
        self.critic1_optimizer = Adam(CRITIC_ALPHA)
        self.critic2_optimizer = Adam(CRITIC_ALPHA)

        self.buffer = ReplayBuffer(10000)

    def build_actor_network(self, state_dim, action_dim):
        inputs = tf.keras.layers.Input(shape=(state_dim,))
        layer1 = tf.keras.layers.Dense(200, activation='relu')(inputs)
        layer2 = tf.keras.layers.Dense(200, activation='relu')(layer1)
        outputs = tf.keras.layers.Dense(action_dim, activation='tanh')(layer2)
        return tf.keras.models.Model(inputs, outputs)

    def build_critic_network(self, state_dim, action_dim):
        state_input = tf.keras.layers.Input(shape=(state_dim,))
        action_input = tf.keras.layers.Input(shape=(action_dim,))
        concat_layer = tf.keras.layers.Concatenate()([state_input, action_input])
        layer1 = tf.keras.layers.Dense(200, activation='relu')(concat_layer)
        layer2 = tf.keras.layers.Dense(200, activation='relu')(layer1)
        outputs = tf.keras.layers.Dense(1)(layer2)
        return tf.keras.models.Model([state_input, action_input], outputs)


    def update_target_networks(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

    def train(self, batch_size):
        # Sample a minibatch from the replay buffer
        batch = self.buffer.sample(batch_size)
        
        if batch is None:
            return None, None, None
        
        states, actions, rewards, next_states, dones = batch

    
        # Update critics
        critic_loss1 = self.update_critic(self.critic1, self.target_critic1, states, actions, rewards, next_states, dones)
        critic_loss2 = self.update_critic(self.critic2, self.target_critic2, states, actions, rewards, next_states, dones)

        # Update actor
        actor_loss = self.update_actor(states)

        return critic_loss1, critic_loss2, actor_loss

    def update_critic(self, critic, target_critic, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_noise = tf.clip_by_value(tf.random.normal(target_actions.shape, stddev=0.1), -0.1, 0.1)
            target_actions += target_noise
            target_actions = tf.clip_by_value(target_actions, env.action_space.low, env.action_space.high)

            # Reshape actions to match critic input shape
            actions = tf.reshape(actions, (-1, self.action_dim))

            target_q_values = critic([next_states, target_actions])
            target_q_values = rewards + GAMMA * target_q_values * (1 - dones)
            current_q_values = critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(target_q_values - current_q_values))

        critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

        return critic_loss



    def update_actor(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q_values = self.critic1([states, actions])
            actor_loss = -tf.reduce_mean(q_values)

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        return actor_loss

    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def act(self, state, noise_factor=0.1):
        action = self.actor.predict(state)
        # print("Action before adding noise:", action)
        action += np.random.normal(0, noise_factor, size=action.shape)
        # print("Action after adding noise:", action)
        # action = np.clip(action, -1, 1)
        # print("Action after clipping:", action)
        return action



class UAVEnvironment:
    def __init__(self, aoi_iotds, scheduled_iotds, current_x, current_y, current_time_block, buffer_size):
        self.propagation_d = PROPAGATION_D
        self.propagation_c = PROPAGATION_C
        self.path_loss_alpha = PATH_LOSS_ALPHA
        self.carrier_frequency = CARRIER_FREQUENCY
        self.noise_power = NOISE_POWER
        self.height = HEIGHT
        self.max_bandwidth = MAX_BANDWIDTH
        self.gamma_future_rewards = GAMMA_FUTURE_REWARDS
        self.los_n1 = LOS_N1
        self.nlos_n2 = NLOS_N2
        self.decay_action_randomness = DECAY_ACTION_RANDOMNESS
        self.gamma = GAMMA
        self.self_replacement_value = SELF_REPLACEMENT_VALUE
        self.m = M
        self.iotd_loc = IOTD_LOC
        self.aoi_iotds = np.zeros_like(aoi_iotds)
        self.scheduled_iotds = scheduled_iotds
        self.current_x = current_x
        self.current_y = current_y
        self.current_time_block = current_time_block
        self.max_data = MAX_DATA
        self.data_to_transmit = np.array([MAX_DATA for _ in range(self.m)])
        self.c = 3e8
        
        self.x_max = X_MAX
        self.y_max = Y_MAX
        self.v = 10
        self.transmission_weight = 1
        self.uav_weight = 0.5
        self.buffer_size = buffer_size
        self.buffer = ReplayBuffer(buffer_size)

        self.path_taken_x = [current_x]
        self.path_taken_y = [current_y]

        # New parameters
        self.h_penalty = 0.1  # Positive constant penalty
        self.h_award = 0.1 # Positive constant reward

        # Blade speed and other parameters
        self.p_uav = 100  # UAV power (in Watts)
        self.Po = 100  # Blade power profile
        self.Pi = 50  # Induced power in the hovering state
        self.Ph = 20  # Hovering energy consumption
        self.do = 0.1  # Fuselage drag ratio
        self.solidity = 0.1  # Blade solidity
        self.rho = 1.225  # Air density (in kg/m^3) - typical value at sea level
        self.area = 1  # Blade area (in m^2)
        self.utip = 100  # Tip speed (in m/s)
        self.v0 = 10  # Mean rotor induced velocity (in m/s)

        self.p = np.array([10 for _ in range(self.m)])

        self.observation_space = spaces.Box(low=np.array([0, 0] + [0] * self.m * 2),
                                            high=np.array([self.x_max, self.y_max] + [10000] * self.m + [1]*self.m),
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([0, self.v, -np.pi]),
                                        high=np.array([1, self.v, np.pi]),
                                        dtype=np.float32)

    def compute_max_distance(self):
        # Calculate distances to each boundary
        distance_left = self.current_x
        distance_right = self.x_max - self.current_x
        distance_bottom = self.current_y
        distance_top = self.y_max - self.current_y
        
        # Return the minimum distance
        return min(distance_left, distance_right, distance_bottom, distance_top)


    def reset(self):
        # self.current_x = np.random.uniform(0, self.x_max)
        # self.current_y = np.random.uniform(0, self.y_max)
        self.current_x = 0
        self.current_y = 0
        self.current_time_block = 0
        # self.iotd_loc = [[random.uniform(0, self.x_max), random.uniform(0, self.y_max)] for _ in range(self.m)]
        self.path_taken_x = [self.current_x]
        self.path_taken_y = [self.current_y]
        self.aoi_iotds = np.zeros_like(self.aoi_iotds)
        self.scheduled_iotds = np.array([1]*self.m)
        self.buffer = ReplayBuffer(self.buffer_size)
        return np.array([self.current_x, self.current_y] + list(self.aoi_iotds) + list(self.scheduled_iotds))

    def step(self, action):
        distance, speed, angle = action


        #some signal calculation
        # d_t = [np.sqrt(self.height ** 2 + (self.iotd_loc[i][0] - self.current_x) ** 2 + (self.iotd_loc[i][1] - self.current_y) ** 2) for i in range(self.m)]
        # O_t = [(180 / np.pi) * np.arcsin(self.height / d) for d in d_t]
        # P_t_LOS = [(1 / (1 + self.c * np.exp(-self.path_loss_alpha * (o - self.propagation_c)))) for o in O_t]
        # b_0 = (4 * np.pi * self.carrier_frequency / self.c) ** (-self.path_loss_alpha)
        # b_t = [P_t_LOS[i] * b_0 * (d_t[i] ** self.path_loss_alpha) / self.los_n1 + (1 - P_t_LOS[i]) * b_0 * (d_t[i] ** self.path_loss_alpha) / self.nlos_n2 for i in range(self.m)]
        # bandwidth_t = [self.scheduled_iotds[i] * self.max_bandwidth / sum(self.scheduled_iotds) for i in range(self.m)]
        # R_t_m = [bandwidth_t[i] * np.log2(1 + (self.p[i] * b_t[i]) / self.noise_power ** 2) for i in range(self.m)]
        # b_t_uav = self.max_bandwidth
        # R_t_uav = b_t_uav * np.log2(1 + (self.p_uav * b_t_uav) / self.noise_power ** 2)
        # D_up_t = max(self.scheduled_iotds[i] * self.data_to_transmit[i] / R_t_m[i] for i in range(self.m))
        # E_t = [self.p[i] * self.data_to_transmit[i] / R_t_m[i] for i in range(self.m)]
        # E_t_IOTD = sum(E_t)

        d_t = [np.sqrt(self.height ** 2 + (self.iotd_loc[i][0] - self.current_x) ** 2 + (self.iotd_loc[i][1] - self.current_y) ** 2) for i in range(self.m)]
        O_t = [(180 / np.pi) * np.arcsin(self.height / d) for d in d_t]
        P_t_LOS = [(1 / (1 + self.c * np.exp(-self.path_loss_alpha * (o - self.propagation_c)))) for o in O_t]
        b_0 = (4 * np.pi * self.carrier_frequency / self.c) ** (-self.path_loss_alpha)
        b_t = [P_t_LOS[i] * b_0 * (d_t[i] ** self.path_loss_alpha) / self.los_n1 + (1 - P_t_LOS[i]) * b_0 * (d_t[i] ** self.path_loss_alpha) / self.nlos_n2 for i in range(self.m)]
        # bandwidth_t = [self.scheduled_iotds[i] * self.max_bandwidth / sum(self.scheduled_iotds) for i in range(self.m)]
        bandwidth_t = [self.max_bandwidth/self.m for i in range(self.m)]
        # Handling division by zero and invalid value encountered
        R_t_m = [bandwidth_t[i] * np.log2(1 + (self.p[i] * b_t[i]) / self.noise_power ** 2) if b_t[i] != 0 else 0 for i in range(self.m)]
        D_up_t = max(self.scheduled_iotds[i] * self.data_to_transmit[i] / (R_t_m[i] + 1) for i in range(self.m))
        E_t = [self.p[i] * self.data_to_transmit[i] / (R_t_m[i] + 1) for i in range(self.m)]
        E_t_IOTD = sum(E_t)


        D_fly_t = distance / speed

        #end here

        E_travel_UAV = distance * (self.Po * (1 + 3 * (self.v / self.utip) ** 2) + self.Pi * self.v0 / self.v + self.do * self.solidity * self.rho * self.area * (self.v ** 3) / 2)
        E_hover_UAV = self.Ph * D_up_t
        E_t_UAV = E_travel_UAV + E_hover_UAV

        self.current_time_block += 1

        

        # distance = min(distance, self.compute_max_distance())

        new_x = self.current_x + distance * np.cos(angle)
        new_y = self.current_y + distance * np.sin(angle)


        new_x = min(max(0, new_x), self.x_max)
        new_y = min(max(0, new_y), self.y_max)

        self.current_x = new_x
        self.current_y = new_y
        # Calculate the penalty term h_penalty and reward term h_award based on scheduling values
        h_penalty = self.h_penalty * sum(self.scheduled_iotds) / len(self.scheduled_iotds) if any(self.scheduled_iotds) else 0
        h_award = self.h_award * (1 - sum(self.scheduled_iotds) / len(self.scheduled_iotds)) if not all(self.scheduled_iotds) else 0
        
        min_distance = 2
        for i in range(self.m):
            distance_to_iotd = np.sqrt((self.current_x - self.iotd_loc[i][0])**2 + (self.current_y - self.iotd_loc[i][1])**2)
            if distance_to_iotd <= min_distance:
                self.scheduled_iotds[i] = 0
                print("IOTD", i, "has been unscheduled")

        for i in range(len(self.aoi_iotds)):
            if self.scheduled_iotds[i] == 1:
                self.aoi_iotds[i] = D_fly_t + D_up_t
            else:
                self.aoi_iotds[i] += (D_fly_t + D_up_t)

        A_t = sum(self.aoi_iotds[i] for i in range(self.m))
        reward = -A_t - self.transmission_weight * E_t_IOTD - self.uav_weight * E_t_UAV - h_penalty + h_award
        
        # New: Adjust the reward to incentivize reducing AoI
        # Penalize higher AoI and reward lower AoI

        
        # Adjust UAV movement to maximize coverage of IoT devices
        # For simplicity, let's assume the UAV moves towards the IoT device with the highest AoI
        max_aoi_index = np.argmax(self.aoi_iotds)
        target_x, target_y = self.iotd_loc[max_aoi_index]

        # Calculate the distance and angle to the target IoT device
        distance = np.sqrt((target_x - self.current_x) ** 2 + (target_y - self.current_y) ** 2)
        angle = np.arctan2(target_y - self.current_y, target_x - self.current_x)

        # Take the minimum of distance and the UAV's maximum movement distance
        distance = min(distance, self.v)
        
        # # Update UAV's position
        new_x = self.current_x + distance * np.cos(angle)
        new_y = self.current_y + distance * np.sin(angle)

        # # Ensure UAV stays within the environment's bounds
        new_x = min(max(0, new_x), self.x_max)
        new_y = min(max(0, new_y), self.y_max)

        # # Update current position
        
        done = self.current_time_block >= LAST_TIME_IOTD

        self.current_x = new_x
        self.current_y = new_y
        
        self.path_taken_x.append(self.current_x)
        self.path_taken_y.append(self.current_y)

        

        done = (sum(self.scheduled_iotds) == 0)
        self.buffer.add((self.current_x, self.current_y, *self.aoi_iotds, *self.scheduled_iotds), action, reward, (new_x, new_y, *self.aoi_iotds, *self.scheduled_iotds), done)

        next_observation = np.array([self.current_x, self.current_y] + list(self.aoi_iotds) + list(self.scheduled_iotds))
        return next_observation, reward, done, {}

    def render(self, mode='human'):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.current_x, self.current_y, color='red', label='UAV')

        for i, (x, y) in enumerate(self.iotd_loc):
            if self.scheduled_iotds[i] == 1:
                plt.scatter(x, y, color='green', label='Scheduled IoT Device')
            else:
                plt.scatter(x, y, color='blue', label='Unscheduled IoT Device')

        # Plot the path the UAV took
        plt.plot(self.path_taken_x, self.path_taken_y, color='orange', label='UAV Path')

        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title('UAV Environment')
        plt.legend()
        plt.grid(True)
        plt.show()

    def close(self):
        pass


aoi_iotds = np.zeros(M)
scheduled_iotds = np.array([1]*M)
current_x = 0
current_y = 0
current_time_block = 0
buffer_size = 10000

env = UAVEnvironment(aoi_iotds, scheduled_iotds, current_x, current_y, current_time_block, buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = TD3Agent(state_dim, action_dim)

max_episodes = 1000
max_steps_per_episode = 1000

for episode in range(1, max_episodes + 1):
    state = env.reset()
    episode_reward = 0
    step = 0
    last_schedule = [1]*M
    # Iterate until all IoT devices are covered
    while sum(env.scheduled_iotds)>0:
        action = agent.act(tf.expand_dims(state, axis=0))
        action = action[0]
        # print(action)
        # if(action[2]<0):
        #     break
        # action = np.clip(action, env.action_space.low, env.action_space.high)
        print(action)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train(batch_size=100)
        state = next_state
        episode_reward += reward
        # if step % 500 == 0:
        #     env.render()
        step += 1
        print(f"Episode: {episode}, Reward: {episode_reward}")
        print(sum(env.scheduled_iotds))
        sched = False
        # for i in range(M):
        #     if env.scheduled_iotds[i] == 0 and last_schedule[i] == 1:
        #         print("IOTD", i, "has been unscheduled")
        #         print("Current UAV positioon:", env.current_x, env.current_y)
        #         print("IOTD position:", env.iotd_loc[i][0], env.iotd_loc[i][1])
        #         env.render()
        # last_schedule = env.scheduled_iotds
        if done or step>=max_steps_per_episode:
            break
    print(sum(env.scheduled_iotds))
    if episode % 10 == 0:
        env.render()    
    if episode_reward > 0 or episode_reward <= 0:
        print(f"Episode: {episode}, Reward: {episode_reward}")
    else:
        print("Error: Episode reward is NaN")
        break

env.close()

import os

def save_model(model, folder_name):
    # Create the directory if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Save the model architecture as JSON
    model_json = model.to_json()
    with open(os.path.join(folder_name, "model_architecture.json"), "w") as json_file:
        json_file.write(model_json)
    
    # Save the model weights
    model.save_weights(os.path.join(folder_name, "model_weights.h5"))

# Call the function to save the actor and critic models
save_model(agent.actor, "Oldies")
save_model(agent.critic1, "Oldies")
save_model(agent.critic2, "Oldies")

