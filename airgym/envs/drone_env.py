import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser
import centerline

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.const_vel_x = 2
        self.const_vel_x_rot = 1.5
        self.const_yaw_rate = 30
        self.centerline_coordinates = centerline.data
        self.iter_count = 0
        self.total_reward = 0
        # loading the mutltirotor client from AirSim
        self.drone = airsim.MultirotorClient(ip=ip_address)
        
        """
        Defining the action space. here we are considering forward
        movement, and left and right turn as 3 actions which will be used
        for training
        """
        
        self.action_space = spaces.Discrete(3)

        # Defining the state, to record observation after each step
        self.obs = {
            "Left_View": np.zeros((84, 84)),
            "Right_View": np.zeros((84, 84)),
        }  

        self.state = {
            'prev_position': np.zeros(3),
            'position': np.zeros(3),
            'Collision': False
        }

        # Setting up the flight scenario
        self._setup_flight()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # take off
        print("start take off")
        self.drone.moveToPositionAsync(0, 0, -2, 3).join()
        print("took off")

    def transform_obs(self, responses):
        # normalize the image and then convert it into [0, 255] scale
        
        try:
            max_pixel_value = np.array(responses[0].image_data_float, dtype=np.float).max()
            print("max Pixel value:", max_pixel_value)
        
            if max_pixel_value != 0:
                img1d = 255 * np.array(responses[0].image_data_float, dtype=np.float)/max_pixel_value
            else:
                img1d = max_pixel_value * np.array(responses[0].image_data_float, dtype=np.float)
            
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

            from PIL import Image

            image = Image.fromarray(img2d)
            im_final = np.array(image.resize((84, 84)))

            return im_final.reshape([84, 84])
        except:
            
            max_pixel_value = 0
            print("Camera exception Occured!!!! assigning all pixels:", max_pixel_value)
            np.savetxt("Exception/Camera_Exception"+str(self.iter_count), responses)

            return np.zeros((84, 84))
        

    def _get_obs(self):
        """
        This Function gets the observation, in our case, the optical flow map from both
        left and the right camera

        self.obs    := stores the left and the right image
        self.state  := stores the previous position, current position and the collison
                       information 
        """
        # Getting the Left Camera Optical Flow Map
        self.image_request_left = airsim.ImageRequest("0", airsim.ImageType.OpticalFlowVis, True, False) 
        response_left = self.drone.simGetImages([self.image_request_left])
        

        # Getting the Right Camera Optical Flow Map
        self.image_request_right = airsim.ImageRequest("1", airsim.ImageType.OpticalFlowVis, True, False)        
        response_right = self.drone.simGetImages([self.image_request_right])
        
        
        image_left = self.transform_obs(response_left)
        
        image_right = self.transform_obs(response_right)
        

        # Updating the observation
        self.obs["Left_View"] = image_left

        self.obs["Right_View"] = image_right
        
        # Updating the states
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position

        'velocity not requried now, maybe in future work'
        # self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        
        self.state["collision"] = self.drone.simGetCollisionInfo().has_collided
        
        # stacking the information from the two cameras to a single image array
        
        image = np.zeros(self.image_shape)
        image[:, 0:84, 0] = self.obs["Left_View"]
        image[:, 84:168, 0] = self.obs["Right_View"]

        return image

    def _do_action(self, action):
        
        print("Current Action:", action)
        self.action = action
        if action == 0:
            self.drone.moveByVelocityZBodyFrameAsync(self.const_vel_x, 0, -2, 0.5, drivetrain = airsim.DrivetrainType.ForwardOnly).join()
        elif action == 1:
            #self.drone.rotateByYawRateAsync(self.const_yaw_rate, 1 ).join()
            self.drone.moveByVelocityZBodyFrameAsync(self.const_vel_x_rot, 0, -2, 0.5, yaw_mode = airsim.YawMode(is_rate = True, yaw_or_rate = -1 * self.const_yaw_rate)).join()
        elif action == 2:
            #self.drone.rotateByYawRateAsync(-1*self.const_yaw_rate, 1 ).join()
            self.drone.moveByVelocityZBodyFrameAsync(self.const_vel_x_rot, 0, -2, 0.5, yaw_mode = airsim.YawMode(is_rate = True, yaw_or_rate = self.const_yaw_rate)).join()
        else:
            self.drone.hoverAsync().join()

    def _compute_reward(self):
        reward_centerline = 0
        reward_target = 0
        reward_distance = 0
        done = 0
        thresh_distance = 0.3           # prev: 0.3             0.1 for narrow corridor
        
        # getting the current position of the multirotor

        rotor_state = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val
                )
            )
        )

        if self.state["collision"]:
            #reward = -10000
            reward = 0
            done = 1
        else:
        # Penalising if the multirotor flies too far(thresh distance) from the centerline of the corridor
        # for each iteration, reward will be computed by evaluating only 6 close centerline coordinates
            distance = 10000000
            
            for i in range(0, self.centerline_coordinates.shape[0] - 1):
                distance = min(
                    distance, np.linalg.norm(np.cross((rotor_state - self.centerline_coordinates[i]), 
                    (rotor_state - self.centerline_coordinates[i + 1])))
                    / np.linalg.norm(self.centerline_coordinates[i]
                    - self.centerline_coordinates[i + 1])
                )
            
            
            '''
            try:
                for i in range(self.iter_count, 6 + self.iter_count):
                    distance = min(
                        distance, np.linalg.norm(np.cross((rotor_state - self.centerline_coordinates[i]), 
                        (rotor_state - self.centerline_coordinates[i + 1])))
                        / np.linalg.norm(self.centerline_coordinates[i]
                         - self.centerline_coordinates[i + 1 ])
                    )
            except:
                    for i in range(self.iter_count, self.centerline_coordinates.shape[0] - 1):
                        distance = min(
                            distance, np.linalg.norm(np.cross((rotor_state - self.centerline_coordinates[i]), 
                            (rotor_state - self.centerline_coordinates[i + 1])))
                            / np.linalg.norm(self.centerline_coordinates[i]
                             - self.centerline_coordinates[i + 1])
                        )
            '''
            # Reward if the multirotor is flying close to the centerline
            if distance > thresh_distance:
                reward_centerline = 1
            else:
                reward_centerline = 3

            # Reward if the multirotor has reached the target location
            distance_from_target = np.exp(np.linalg.norm(rotor_state - self.centerline_coordinates[-1]))
            if distance_from_target <= 0.1:
                reward_target = 10000
                done = 1
                print("------------------------------- Reached Target -----------------------------")
            else:
                reward_distance = 10000 / distance_from_target

            reward = reward_target + reward_centerline + reward_distance
            '''
            print("------------------------------------")
            print("Distance Reward: ", reward_distance)
            print("Centerline Reward: ", reward_centerline)
            print("Target Reward: ", reward_target)
            print('------------------------------------')
            self.iter_count += 1
            '''
        return reward, done

    def step(self, action):
        self.iter_count += 1
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        #print("Current Action Reward: ", reward)
        self.total_reward += reward
        if done:
            print("##############################-- Game Over --##################################")
            print("Total Reward: ", self.total_reward)
            print('-------------------------------------------------------------------------------')
            print("###############################################################################")

        #np.savetxt("optical_flow_left", self.state["Left_View"])

        return obs, reward, done, self.state

    def reset(self):
        self.total_reward = 0
        self.iter_count = 0
        self._setup_flight()
        return self._get_obs()

