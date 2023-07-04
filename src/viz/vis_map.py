import os
import json
import cv2
import argparse
import numpy as np
import random
from collections import deque

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
    print(white_pixels_surrounded_by_black)
    for y, x in white_pixels_surrounded_by_black:
        print([y, x])
        image[y, x] = (0, 0, 0)
        print(image[y,x])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cv2.imwrite(save_path + "/" + file_name + '.png', image)
# def draw_map(file_name, json_path, save_path):
#     print("Processing ", file_name)
#
#     with open(json_path + '\\' + file_name + '.json') as json_file:
#         json_data = json.load(json_file)
#
#     # Draw the contour
#     verts = (np.array(json_data['verts']) * meter2pixel).astype(int)
#
#
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the subset of maps in .png.")

    parser.add_argument("--map_id_set_file", help="map id set (.txt)",
                        default=r'..\..\assets\a.txt')
    parser.add_argument("--json_path", type=str, default=r'..\..\assets\json',
                        help="json file path")
    parser.add_argument("--save_path", type=str, default=r'..\..\assets\png')


    result = parser.parse_args()

    json_path = os.path.abspath(os.path.join(os.getcwd(), result.json_path))
    map_file = os.path.abspath(os.path.join(os.getcwd(), result.map_id_set_file))
    save_path = os.path.abspath(os.path.join(os.getcwd(), result.save_path))
    print("---------------------------------------------------------------------")
    print("|map id set file path        |{}".format(map_file))
    print("---------------------------------------------------------------------")
    print("|json file path              |{}".format(json_path))
    print("---------------------------------------------------------------------")
    print("|Save path                   |{}".format(save_path))
    print("---------------------------------------------------------------------")

    map_ids = np.loadtxt(map_file, str)

    if map_ids.shape == ():
        map_ids = np.reshape(map_ids, (1,))
    for map_id in map_ids:
        draw_map(map_id, save_path)


    print("Successfully draw the maps into {}.".format(save_path))