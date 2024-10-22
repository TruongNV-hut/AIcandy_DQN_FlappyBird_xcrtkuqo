from itertools import cycle
from numpy.random import randint
from pygame import Rect, init, time, display
from pygame.event import pump
from pygame.image import load
from pygame.surfarray import array3d, pixels_alpha
from pygame.transform import rotate
import numpy as np
import pygame
import os
from datetime import datetime
from time import sleep
import logging

class FlappyBird(object):
    init()
    game_clock = time.Clock()
    display_width = 288
    display_height = 512
    game_display = display.set_mode((display_width, display_height))
    display.set_caption('AIcandy.vn Flappy Bird')
    ground_img = load('aicandy_data/base.png').convert_alpha()
    bg_img = load('aicandy_data/background-black.png').convert()

    obstacle_imgs = [rotate(load('aicandy_data/pipe-green.png').convert_alpha(), 180),
                     load('aicandy_data/pipe-green.png').convert_alpha()]
    player_imgs = [load('aicandy_data/redbird-upflap.png').convert_alpha(),
                   load('aicandy_data/redbird-midflap.png').convert_alpha(),
                   load('aicandy_data/redbird-downflap.png').convert_alpha()]

    player_mask = [pixels_alpha(image).astype(bool) for image in player_imgs]
    obstacle_mask = [pixels_alpha(image).astype(bool) for image in obstacle_imgs]

    frame_rate = 30
    obstacle_gap = 100
    obstacle_speed = -4

    min_vert_speed = -8
    max_vert_speed = 10
    gravity = 1
    flap_power = -9

    player_anim_cycle = cycle([0, 1, 2, 1])

    def __init__(self):
        self.frame_count = self.player_frame = self.points = 0

        self.player_width = self.player_imgs[0].get_width()
        self.player_height = self.player_imgs[0].get_height()
        self.obstacle_width = self.obstacle_imgs[0].get_width()
        self.obstacle_height = self.obstacle_imgs[0].get_height()

        self.player_x = int(self.display_width / 5)
        self.player_y = int((self.display_height - self.player_height) / 2)

        self.ground_x = 0
        self.ground_y = self.display_height * 0.79
        self.ground_shift = self.ground_img.get_width() - self.bg_img.get_width()

        obstacles = [self.create_obstacle(), self.create_obstacle()]
        obstacles[0]["x_top"] = obstacles[0]["x_bottom"] = self.display_width
        obstacles[1]["x_top"] = obstacles[1]["x_bottom"] = self.display_width * 1.5
        self.obstacles = obstacles

        self.vert_speed = 0
        self.flapped = False
        self.movement_index = 0

    def create_obstacle(self):
        x = self.display_width + 10
        gap_y = randint(2, 10) * 10 + int(self.ground_y / 5)
        return {"x_top": x, "y_top": gap_y - self.obstacle_height, "x_bottom": x, "y_bottom": gap_y + self.obstacle_gap}

    def check_collision(self):
        if self.player_height + self.player_y + 1 >= self.ground_y:
            return True
        player_rect = Rect(self.player_x, self.player_y, self.player_width, self.player_height)
        obstacle_rects = []
        for obstacle in self.obstacles:
            obstacle_rects.append(Rect(obstacle["x_top"], obstacle["y_top"], self.obstacle_width, self.obstacle_height))
            obstacle_rects.append(Rect(obstacle["x_bottom"], obstacle["y_bottom"], self.obstacle_width, self.obstacle_height))
            if player_rect.collidelist(obstacle_rects) == -1:
                return False
            for i in range(2):
                overlap_rect = player_rect.clip(obstacle_rects[i])
                x1 = overlap_rect.x - player_rect.x
                y1 = overlap_rect.y - player_rect.y
                x2 = overlap_rect.x - obstacle_rects[i].x
                y2 = overlap_rect.y - obstacle_rects[i].y
                if np.any(self.player_mask[self.player_frame][x1:x1 + overlap_rect.width,
                       y1:y1 + overlap_rect.height] * self.obstacle_mask[i][x2:x2 + overlap_rect.width,
                                                              y2:y2 + overlap_rect.height]):
                    return True
        return False

    def update_game(self, action):
        pump()
        reward = 0.1
        game_over = False
        if action == 1:
            self.vert_speed = self.flap_power
            self.flapped = True

        # Update score
        player_center_x = self.player_x + self.player_width / 2
        for obstacle in self.obstacles:
            obstacle_center_x = obstacle["x_top"] + self.obstacle_width / 2
            if obstacle_center_x < player_center_x < obstacle_center_x + 5:
                self.points += 1
                reward = 1
                break

        if (self.frame_count + 1) % 3 == 0:
            self.player_frame = next(self.player_anim_cycle)
            self.frame_count = 0
        self.ground_x = -((-self.ground_x + 100) % self.ground_shift)

        if self.vert_speed < self.max_vert_speed and not self.flapped:
            self.vert_speed += self.gravity
        if self.flapped:
            self.flapped = False
        self.player_y += min(self.vert_speed, self.player_y - self.vert_speed - self.player_height)
        if self.player_y < 0:
            self.player_y = 0

        for obstacle in self.obstacles:
            obstacle["x_top"] += self.obstacle_speed
            obstacle["x_bottom"] += self.obstacle_speed

        if 0 < self.obstacles[0]["x_bottom"] < 5:
            self.obstacles.append(self.create_obstacle())
        if self.obstacles[0]["x_bottom"] < -self.obstacle_width:
            del self.obstacles[0]
        if self.check_collision():
            game_over = True
            reward = -1

        self.game_display.blit(self.bg_img, (0, 0))
        self.game_display.blit(self.ground_img, (self.ground_x, self.ground_y))
        self.game_display.blit(self.player_imgs[self.player_frame], (self.player_x, self.player_y))
        for obstacle in self.obstacles:
            self.game_display.blit(self.obstacle_imgs[0], (obstacle["x_top"], obstacle["y_top"]))
            self.game_display.blit(self.obstacle_imgs[1], (obstacle["x_bottom"], obstacle["y_bottom"]))
        frame = array3d(display.get_surface())
        self.display_score()
        display.update()
        self.game_clock.tick(self.frame_rate)
        return frame, reward, game_over

    def display_score(self):
        try:
            font_size = 30
            font = pygame.font.SysFont('Tahoma', font_size)
            score_text = str(int(self.points))
            score_surface = font.render(score_text, True, (255,0,0))
            text_x = int(self.bg_img.get_width() - 5*len(score_text))/2 - 20
            text_y = int(self.bg_img.get_height() - 60)
            self.game_display.blit(score_surface, dest=(text_x, text_y))
        except Exception as e:
            logging.exception('An error occurred in display_score function\n')

    def get_score(self):
        return self.points