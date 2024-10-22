import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn

from aicandy_core.model_xpuihdim import DeepQNetwork
from aicandy_core.env_FlappyBird_train_gsbdrvvp import FlappyBird
from aicandy_core.utils_dylhxpln import process_image
import logging
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(
        """AIcandy.vn Flappy Bird""")
    parser.add_argument("--img_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_count", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer_type", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--start_epsilon", type=float, default=0.1)
    parser.add_argument("--end_epsilon", type=float, default=1e-4)
    parser.add_argument("--total_iterations", type=int, default=2500000)
    parser.add_argument("--memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--model_dir", type=str, default="models")

    args = parser.parse_args()
    return args

def train_flappy_bird(config):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    
    network = DeepQNetwork()
    
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    
    environment = FlappyBird()
    frame, reward, done = environment.update_game(0)
    frame = process_image(frame[:environment.display_width, :int(environment.ground_y)], config.img_size, config.img_size)
    frame = torch.from_numpy(frame)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    frame = frame.to(device)
    
    current_state = torch.cat(tuple(frame for _ in range(4)))[None, :, :, :]
    experience_buffer = []
    iteration = 0
    
    while iteration < config.total_iterations:
        prediction = network(current_state)[0]
        epsilon = config.end_epsilon + (
                (config.total_iterations - iteration) * (config.start_epsilon - config.end_epsilon) / config.total_iterations)
        
        if random() <= epsilon:
            action = randint(0, 1)
        else:
            action = torch.argmax(prediction).item()
        
        print(f'Iteration: {iteration + 1}, Action: {action}, Score: {environment.get_score()}')
        
        next_frame, reward, done = environment.update_game(action)
        next_frame = process_image(next_frame[:environment.display_width, :int(environment.ground_y)], config.img_size, config.img_size)
        next_frame = torch.from_numpy(next_frame).to(device)
        next_state = torch.cat((current_state[0, 1:, :, :], next_frame))[None, :, :, :]
        
        experience_buffer.append([current_state, action, reward, next_state, done])
        if len(experience_buffer) > config.memory_size:
            del experience_buffer[0]
        
        batch = sample(experience_buffer, min(len(experience_buffer), config.batch_count))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))
        
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)
        
        current_q_values = network(state_batch)
        next_q_values = network(next_state_batch)

        target_q_values = torch.cat(
            tuple(reward if done else reward + config.discount_factor * torch.max(q_value) for reward, done, q_value in
                  zip(reward_batch, done_batch, next_q_values)))

        q_value = torch.sum(current_q_values * action_batch, dim=1)
        optimizer.zero_grad()
        loss = loss_function(q_value, target_q_values)
        loss.backward()
        optimizer.step()

        current_state = next_state
        iteration += 1
        
        if (iteration + 1) % 20000 == 0:
            torch.save(network, f"{config.model_dir}/model_FlappyBird_{iteration+1}")
            try:
                current_time = datetime.now()
                with open('logs/AAA_logTimeTrain.txt', 'a') as f:
                    f.write(f"{iteration+1} DateTime: {current_time}\n")
            except Exception as e:
                logging.exception('An error occurred while logging\n')

    torch.save(network, f"{config.model_dir}/model_FlappyBird_{config.total_iterations}")

if __name__ == "__main__":
    config = parse_arguments()
    train_flappy_bird(config)