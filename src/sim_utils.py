import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
import json
import cv2
import argparse
import random
from collections import deque

meter2pixel = 100
border_pad = 25

CONFIG_PATH = os.getcwd()+'/../assets/config.ymal'

def within_bound(p,shape,r=0):
    """ check if point p [y;x] or [y;x;theta] with radius r is inside world of shape (h,w)
    return bool if p is single point | return bool matrix (vector) if p: [y;x] where y & x are matrix (vector) """
    return (p[0] >= r) & (p[0] < shape[0]-r) & (p[1] >= r) & (p[1] < shape[1]-r)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def onehot_from_action(actions):
    onehot = np.zeros((len(actions),8))
    for i,action in enumerate(actions):
        onehot[i,action] = 1
    return onehot


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


# def draw_map(file_name, json_path, save_path):
#     """
#     generate map picture of the "file_name.json", and save it into save_path
#     :param file_name:
#     :param json_path:
#     :param save_path:
#     :return: None
#     """
#     meter2pixel = 100
#     border_pad = 25
#     print("Processing ", file_name)
#     with open(json_path + '/' + file_name + '.json') as json_file:
#         json_data = json.load(json_file)
#
#     # Draw the contour
#     verts = (np.array(json_data['verts']) * meter2pixel).astype(np.int)
#     x_max, x_min, y_max, y_min = np.max(verts[:, 0]), np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 1])
#     cnt_map = np.zeros((y_max - y_min + border_pad * 2,
#                         x_max - x_min + border_pad * 2))
#
#     verts[:, 0] = verts[:, 0] - x_min + border_pad
#     verts[:, 1] = verts[:, 1] - y_min + border_pad
#     cv2.drawContours(cnt_map, [verts], 0, 255, -1)
#
#     # Save map
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     cv2.imwrite(save_path + "/" + file_name + '.png', cnt_map)
#
#
# def draw_maps(map_ids, json_path, save_path):
#     json_path = os.path.join(os.getcwd(),json_path)
#     save_path = os.path.join(os.getcwd(),save_path)
#     for map_id in map_ids:
#         draw_map(map_id,json_path,save_path)
#     print('Draw the map successfully.')


def draw_maze_map(file_name, save_path):
    maze = np.zeros((20, 20, 3), dtype=np.uint8)

    # Set the starting position
    start_x, start_y = random.randint(0, 19), random.randint(0, 19)
    maze[start_y, start_x] = (255, 255, 255)

    # Perform depth-first search to create the maze
    stack = [(start_x, start_y)]

    while stack:
        x, y = stack[-1]

        # Get the unvisited neighbors
        neighbors = [(x - 2, y), (x + 2, y), (x, y - 2), (x, y + 2)]
        unvisited_neighbors = [neighbor for neighbor in neighbors if
                               0 <= neighbor[0] < 20 and 0 <= neighbor[1] < 20 and all(
                                   maze[neighbor[1], neighbor[0]] == 0)]
        if unvisited_neighbors:
            nx, ny = random.choice(unvisited_neighbors)
            maze[ny, nx] = (255, 255, 255)
            maze[ny + (y - ny) // 2, nx + (x - nx) // 2] = (255, 255, 255)
            stack.append((nx, ny))
        else:
            stack.pop()

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cv2.imwrite(save_path + "/" + file_name + '.png', maze)


def is_valid_pixel(image, x, y):
    # Returns true if pixel is within the image bounds and is white
    return 0 <= x < image.shape[0] and 0 <= y < image.shape[1] and np.all(image[y][x] == [255, 255, 255])


def is_surrounded_by_black(image, x, y):
    # Defines possible movement directions (up, down, left, right)
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # Check if all surrounding pixels are black
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if nx >= 0 and ny >= 0 and nx < image.shape[0] and ny < image.shape[1]:
            if np.any(image[ny][nx] != [0, 0, 0]):
                return False
    return True


def get_white_pixels_surrounded_by_black(image):
    visited = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    white_pixels_surrounded_by_black = []

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if is_valid_pixel(image, x, y) and not visited[y][x]:
                queue = deque([(x, y)])
                visited[y][x] = True

                while queue:
                    x, y = queue.popleft()
                    if is_surrounded_by_black(image, x, y):
                        white_pixels_surrounded_by_black.append((x, y))

                    # Check neighboring pixels (up, down, left, right)
                    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        nx, ny = x + dx, y + dy
                        if is_valid_pixel(image, nx, ny) and not visited[ny][nx]:
                            queue.append((nx, ny))
                            visited[ny][nx] = True
    return white_pixels_surrounded_by_black


def draw_map(file_name, save_path):
    image = np.ones((20, 20, 3), dtype=np.uint8) * 255

    # Generate random black dots
    num_dots = 100  # Adjust the range as per your preference

    dot_region_size = 5  # Size of the region to place each dot

    for _ in range(num_dots):
        # Randomly select a region to place the dot
        region_x = random.randint(1, 20 - dot_region_size - 1)
        region_y = random.randint(1, 20 - dot_region_size - 1)

        # Randomly choose a position within the selected region
        x = random.randint(region_x, region_x + dot_region_size - 1)
        y = random.randint(region_y, region_y + dot_region_size - 1)

        # Check if the dot is surrounded by black dots or placed on the boundary
        if (np.all(image[[y - 1, y + 1, y, y], [x, x, x - 1, x + 1]] == 0) or
                x == 0 or x == 19 or y == 0 or y == 19):
            image[y, x] = (0, 0, 0)
            continue

        # Set the pixel color to black
        image[y, x] = (0, 0, 0)
    white_pixels_surrounded_by_black = get_white_pixels_surrounded_by_black(image)
    # print(white_pixels_surrounded_by_black)
    for y, x in white_pixels_surrounded_by_black:
        # print([y, x])
        image[y, x] = (0, 0, 0)
        # print(image[y,x])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cv2.imwrite(save_path + "/" + file_name + '.png', image)
    # map_loader(file_name,save_path)


def draw_maps(map_file, json_path, save_path):
    # map_ids = np.loadtxt(map_file, str)
    #
    # if map_ids.shape == ():
    #     map_ids = np.reshape(map_ids, (1,))
    for map_id in map_file:
        draw_map(map_id, save_path)
    print("Successfully draw the maps into {}.".format(save_path))