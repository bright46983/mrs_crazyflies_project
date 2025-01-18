#!/usr/bin/env python3

import numpy as np
from scipy.ndimage import morphology
from matplotlib import pyplot as plt

import yaml
import cv2

from utils.OccupancyMap import OccupancyMap

def dilate_obstacles(grid, dilation_length):
    obstacle_mask = (grid >= 80)
    structuring_element = np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]], dtype=bool)
    # structuring_element = np.array([[0, 0, 0],
    #                                 [0, 1, 0],
    #                                 [0, 0, 0]], dtype=bool)
    dilated_mask = morphology.binary_dilation(obstacle_mask, iterations=dilation_length, structure=structuring_element)
    dilated_grid = np.zeros_like(grid)
    dilated_grid[dilated_mask] = 100  # Set occupied cells to 100 in the dilated grid
    return dilated_grid


def load_map(yaml_path, image_path):
    with open(yaml_path, 'r') as file:
        map_data = yaml.safe_load(file)
    
    resolution = map_data['resolution']
    origin = map_data['origin']
    negate = map_data['negate']
    
    # Read and optionally negate the map image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if not negate:
        image = 255 - image

    # Flip the image vertically
    image = np.flipud(image)

    # Convert the image to an occupancy grid (-1: unknown, 0: free, 100: occupied)
    occupancy_map = (image / 255.0).flatten()
    occupancy_map = np.where(occupancy_map > map_data['occupied_thresh'], 100,
                            np.where(occupancy_map < map_data['free_thresh'], 0, -1))

    return occupancy_map

def main():
    map_yaml_path = '/home/leopt4/CrazySim/ros2_ws/src/mrs_crazyflies_project/resource/maps/test_10x10/test_10x10.yaml'
    map_image_path = '/home/leopt4/CrazySim/ros2_ws/src/mrs_crazyflies_project/resource/maps/test_10x10/test_10x10.bmp'
    occupancy_map = load_map(map_yaml_path, map_image_path)

    map = OccupancyMap()
    env = np.array(occupancy_map.data).reshape(200, 200).T
    env_fil = env
    # Set avoid obstacles - The steer to avoid behavior (IN THE DICTIONARY) requires the map, resolution, and origin
    map.set(data=env_fil, 
            resolution=0.05, 
            origin=[-5, -5])
    
    # Plot the array
    print(f"Map shape: {map.map.shape}")  # Check the shape of the map
    plt.imshow(map.map, cmap='gray', interpolation='nearest')
    plt.colorbar()  # Optional: Add a color bar to indicate the scale
    plt.title('Array Visualization')
    plt.show()

if __name__ == "__main__":
    main()


