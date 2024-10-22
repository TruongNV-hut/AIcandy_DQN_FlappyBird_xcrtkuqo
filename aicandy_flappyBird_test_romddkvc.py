import argparse
import torch
from aicandy_core.model_xpuihdim import DeepQNetwork
from aicandy_core.env_FlappyBird_test_vhnldpii import FlappyBird
from aicandy_core.utils_dylhxpln import process_image
import shutil
import os
import time
from datetime import datetime
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(
        """AIcandy.vn Flappy Bird""")
    parser.add_argument("--img_dim", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--model_dir", type=str, default="aicandy_models")
    args = parser.parse_args()
    return args

def run_flappy_bird_test(config):
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(f"{config.model_dir}/model_FlappyBird" if torch.cuda.is_available() else f"{config.model_dir}/model_FlappyBird", map_location=device)
        model.eval()
        
        game = FlappyBird()
        frame, reward, done = game.update_game(0)
        frame = process_image(frame[:game.display_width, :int(game.ground_y)], config.img_dim, config.img_dim)
        frame = torch.from_numpy(frame).to(device)
        state = torch.cat(tuple(frame for _ in range(4)))[None, :, :, :]

        while not game.check_collision():
            prediction = model(state)[0]
            action = torch.argmax(prediction).item()
            next_frame, reward, done = game.update_game(action)
            next_frame = process_image(next_frame[:game.display_width, :int(game.ground_y)], config.img_dim, config.img_dim)
            next_frame = torch.from_numpy(next_frame).to(device)
            next_state = torch.cat((state[0, 1:, :, :], next_frame))[None, :, :, :]
            state = next_state
                
        time.sleep(10)  
    except Exception as e:
        logging.exception('An error occurred in run_flappy_bird_test function')

if __name__ == "__main__":
    start_time = datetime.now()
    start_timestamp = int(start_time.timestamp())
    
    config = parse_arguments()
    run_flappy_bird_test(config)
    
    end_timestamp = int(datetime.now().timestamp())
    execution_time = end_timestamp - start_timestamp
    print(f'Program execution time: {execution_time}s') 
    print("--------------------- Finish -----------------------")
    print('\n\n')