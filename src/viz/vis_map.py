import os
import json
import cv2
import argparse
import numpy as np
import random
from collections import deque
from datetime import datetime

meter2pixel = 100
border_pad = 25

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
    num_dots = 20  # Adjust the range as per your preference

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


def map_loader(file_name, save_path, padding=5):
    obstacle = 255
    free = 0
    map_raw = cv2.imread(save_path + "/" + file_name + '.png',cv2.IMREAD_GRAYSCALE)
    maze=np.zeros_like(map_raw)
    maze[map_raw==0] = obstacle
    maze[map_raw==255] = free
    index = np.where(maze==obstacle)
    [index_row_max,index_row_min,index_col_max,index_col_min] = [np.max(index[0]),np.min(index[0]),np.max(index[1]),np.min(index[1])]
    maze = maze[index_row_min:index_row_max+1,index_col_min:index_col_max+1]
    maze = np.lib.pad(maze, padding, mode='constant', constant_values=obstacle)
    maze = cv2.resize(maze,(200,200),interpolation=cv2.INTER_NEAREST)
    # map = cv2.dilate(map, np.ones((3, 3)), iterations=2)
    cv2.imwrite(save_path + "/" + file_name + 'loaded' + '.png', maze)
    return maze

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the subset of maps in .png.")

    parser.add_argument("--save_map_id_set_file", help="map id set (.txt)",
                        default=r'.\assets\map_id.txt')
    parser.add_argument("--pic_num", help="number of pictures",
                        default=100)
    parser.add_argument("--save_path", type=str, default=r'.\assets\png')


    result = parser.parse_args()

    # json_path = os.path.abspath(os.path.join(os.getcwd(), result.json_path))
    # map_file = os.path.abspath(os.path.join(os.getcwd(), result.map_id_set_file))

    map_id_file_path = os.path.abspath(os.path.join(os.getcwd(), result.save_map_id_set_file))
    pic_num = result.pic_num
    save_path = os.path.abspath(os.path.join(os.getcwd(), result.save_path))
    print("---------------------------------------------------------------------")
    print("|map id set file path        |{}".format(map_id_file_path))
    print("---------------------------------------------------------------------")
    print("|Save path                   |{}".format(save_path))
    print("---------------------------------------------------------------------")
    for filename in os.listdir(save_path):
        file_path = os.path.join(save_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    map_id_file = open(map_id_file_path, "w")



    for iteration in range(pic_num):


        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        map_id_file.write("image"+str(iteration)+"-"+current_time+"\n")
        draw_maze_map("image"+str(iteration)+"-"+current_time, save_path)

    map_id_file.close()
    print("Successfully draw the maps into {}.".format(save_path))