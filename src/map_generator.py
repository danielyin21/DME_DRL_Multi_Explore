# map_generator generates all the json files under the directory ../HouseExpo/json

# draw_maps in sim_utils.py takes in train set map ids or test set map ids
# it reads json files under directory: ../HouseExpo/json and then saves them under the directory: ../assets/png
# then it calls draw_map for each map_id

# draw_map reads the "verts" entry oof the json file, and inputs it into the function cv2.drawContours().
# Then it outputs cnt_map and save it under the png diirectory

# map loader takes in an png file and load it.
# the png file is under ../assets/png directory, with file name map_id, and format .png
import os, cv2
import numpy as np
import json, yaml, random

def map_generator(save_path, set_name="train", map_id=None):
    # create a dictionary
    data = {
        "verts": {

        }
    }

    # Read image
    # image = cv2.imread('image.jpg')
    #
    # # Convert image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # Apply thresholding or other image processing operations if needed
    #
    # # Find contours
    # contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # save under directory
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if map_id == None:
        map_id = choose_map_id(set_name)

    file_path = save_path + "/" + map_id + ".json"

    # save in json
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)

def choose_map_id(set_name="train", json_path=os.getcwd()+'/../HouseExpo/json', config_path=os.getcwd()+'/../assets/config.yaml'):
    with open(config_path) as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)

    if set_name == "train":
        map_ids = np.loadtxt(os.getcwd() + config['map_id_train_set'], str)
    else:
        map_ids = np.loadtxt(os.getcwd() + config['map_id_eval_set'], str)

    file_list = os.listdir(json_path)
    is_contained = True

    while is_contained:
        index = random.randint(0, len(map_ids))
        file_name = str(map_ids[index]) + '.json'
        if file_name not in file_list:
            is_contained = False

    return map_ids[index]
