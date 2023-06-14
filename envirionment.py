#!/usr/bin/env python
import math
import random

import carla
from gym.spaces import Box
import gym
from gym import spaces
import numpy as np
from shapely.geometry import Polygon, Point

import connectionCarla
from config_file import var_recording, var_imitation, var_startpos, var_lrl
from walker_vehicle import WalkerClass, VehicleClass
import datetime


# This function calculates the Euclidean distance between two points (pos1 and pos2) in 2D space
def dist_xy(pos1, pos2):
    return math.sqrt(
        (pos1[0] - pos2[0]) ** 2 +
        (pos1[1] - pos2[1]) ** 2
    )


# This function takes a 3D velocity vector (x, y, z) as input and returns its magnitude in meters per second (m/s).
def velocity_3d_to_ms(velocity3d):
    return math.sqrt(velocity3d.x ** 2 + velocity3d.y ** 2 + velocity3d.z ** 2)


def calc_bounding_box(p0, angle, length, width):
    if abs(length) < 0.1:
        length = 0.1
    x = p0[0] + length * math.cos(angle)
    y = p0[1] + length * math.sin(angle)
    p_mid_ahead = (x, y)

    x = p0[0] + width * math.cos(angle - math.pi / 2)
    y = p0[1] + width * math.sin(angle - math.pi / 2)
    point1 = (x, y)

    x = p0[0] + width * math.cos(angle + math.pi / 2)
    y = p0[1] + width * math.sin(angle + math.pi / 2)
    point2 = (x, y)

    x = p_mid_ahead[0] + width * math.cos(angle - math.pi / 2)
    y = p_mid_ahead[1] + width * math.sin(angle - math.pi / 2)
    point3 = (x, y)

    x = p_mid_ahead[0] + width * math.cos(angle + math.pi / 2)
    y = p_mid_ahead[1] + width * math.sin(angle + math.pi / 2)
    point4 = (x, y)
    return Polygon([point1, point2, point4, point3])


def get_current_timestamp():
    x = datetime.datetime.now()
    return f"{x.year}_{x.month}_{x.day}_{x.hour}_{x.minute}"


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, time_steps_per_training=128,
                 render=True, syncMode=True, spawnInfo=[],
                 reloadMap=True, camaraLocation=carla.Location(x=-87.028145, y=2.666814, z=30.478775),
                 camRotation=False, port=2000):
        super(CustomEnv, self).__init__()
        # === RL Varialbles ====
        self.done = False
        self.reward = 0
        self.tick_count = 0
        self.max_tick_count = time_steps_per_training
        self.info = {"actions": []}
        self.actionsList = []
        self.collisionReward = 0
        # 1. Distance, 2. TTC, 3. Angle 4. Dist Imitation 5. Min TCC
        self.rewardDistribution = var_lrl["rewardDistribution"]

        self.angleRewards = []
        self.distenceRewards = []
        self.ttcRewards = []
        self.minttcRewards = []
        self.walkerdistRewards = []
        self.collisionRewards = []
        self.crosswalkRewards = []
        self.rewardGesamt = 0
        # === TTC Calculation ===
        self.timeLookAhead = 3
        self.lengthWalker = 0.187679 * 1.5
        self.widthWalker = 0.187679
        self.lengthVehicle = 2.395890
        self.widthVehicle = 1.081725
        self.lastTTC = []
        self.punishmentTTC = 3
        self.min_ttc = 3
        self.basicReward = 0
        self.lastBasicReward = -999
        self.lastmin_ttc = 999
        self.highest_basic_reward = -999
        self.min_ttc_env = 999
        # Imitation Variables
        self.positions_walker = var_recording["positions_walker"]
        self.maxPunishmentDistance_imitation = var_imitation["maxPunishmentDistance"]

        # === Gym Variables ===
        high_action = np.array([
            1, 1,  # direction vector (x/y)
            1  # speed from 0 to 100%
        ])
        low_action = np.array([
            -1, -1,  # dir_vec (x/y)
            0.01  # speed from 0 to 100%
        ])
        self.action_space = Box(low=low_action, high=high_action, shape=(3,), dtype=np.float32)
        max_x_y_cords = var_startpos["max_x_y_cords"]
        min_x = max_x_y_cords[0][0]
        max_x = max_x_y_cords[0][1]
        min_y = max_x_y_cords[1][0]
        max_y = max_x_y_cords[1][1]

        high = np.array([
            max_x, max_y,  # pos_car(x,y)
            30, 30,  # vel_car(x,y)
            max_x, max_y,  # pos_walker(x,y)
            15, 15,  # vel_walker(x,y)
            1, 1,  # direction vector walker (x, y)
            max_x, max_y  # position walker found situation
        ])
        low = np.array([
            min_x, min_y,  # pos_car(x,y)
            0, 0,  # vel_car(x,y)
            min_x, min_y,  # pos_walker(x,y)
            0, 0,  # vel_walker(x,y)
            -1, -1,  # direction vector walker (x, y)
            min_x, min_y  # pos_walker(x,y)
        ])
        obs_size = len(high)
        self.observation_space = spaces.Box(low=low, high=high,
                                            shape=(obs_size,), dtype=np.float32)

        # === Carla ===
        self.connection = connectionCarla.CarlaConnection()
        self.connection.__int__(render=render, syncMode=syncMode,
                                camaraLocation=camaraLocation,
                                reloadWorld=reloadMap, camRotation=camRotation, port=port)

        # === Walker and Vehicle ===

        spawnLocationWalker = spawnInfo[0]
        spawnLocationVehicle = spawnInfo[1]
        rotationVehicle = spawnInfo[2]
        rotationWalker = spawnInfo[3]
        self.maxPunishmentDistance = dist_xy([spawnLocationWalker.x, spawnLocationWalker.y],
                                             [spawnLocationVehicle.x, spawnLocationVehicle.y]) * 2

        self.connection.draw_waypoint(location=spawnLocationWalker, index='w', life_time=200, intensity=1)
        self.connection.draw_waypoint(location=spawnLocationVehicle, index='v', life_time=200, intensity=1)

        self.walker = WalkerClass(self.connection.world, spawnPoint_init=spawnLocationWalker, rotation=rotationWalker)

        self.vehicle = VehicleClass(self.connection, spawnPoint_init=spawnLocationVehicle,
                                    rotation=rotationVehicle, environment=self)

        self.connection.world.tick()
        # Some other stuff
        self.path_basescore = None
        # self.renderPolys()
        # self.drawsImitationPoints()

    def drawsImitationPoints(self):
        for p in self.positions_walker:
            l = carla.Location(x=p[0], y=p[1], z=8)
            self.connection.draw_waypoint(location=l, index='x', life_time=3000, intensity=0)

    def renderPolys(self):

        point1 = carla.Location(x=159.5, y=-80, z=8)
        point2 = carla.Location(x=170, y=-80, z=8)
        point3 = carla.Location(x=170, y=-132, z=8)
        point4 = carla.Location(x=159.5, y=-132, z=8)
        ps = [point1, point2, point3, point4]

        point5 = carla.Location(x=160, y=-124, z=8)
        point6 = carla.Location(x=160, y=-120, z=8)
        point7 = carla.Location(x=143, y=-120, z=8)
        point8 = carla.Location(x=143, y=-124, z=8)
        ps2 = [point5, point6, point7, point8]

        for i, p in enumerate(ps):
            self.connection.draw_waypoint(p, i, life_time=3000.0, intensity=1)
        for i, p in enumerate(ps2):
            self.connection.draw_waypoint(p, i + 5, life_time=3000.0, intensity=0.5)

    # Renders the environment in either "human" or "machine" mode, enabling or disabling the rendering accordingly.
    def render(self, mode="human"):
        if mode == "human":
            settings = self.connection.world.get_settings()
            settings.no_rendering_mode = False
            self.connection.world.apply_settings(settings)
        else:
            settings = self.connection.world.get_settings()
            settings.no_rendering_mode = True
            self.connection.world.apply_settings(settings)

    # Takes an action, applies it to the walker, performs a world tick, and returns the updated observation, reward,
    # 'done' status, and additional information.
    def step(self, action):
        self.actionsList.append(action)
        self.walker.apply_control(action=action)
        # === Do a tick, an check if done ===
        self.connection.world.tick()
        self.tick_count += 1
        self.vehicle.tick_count += 1
        if self.done and self.tick_count < self.max_tick_count:
            self.saveActionList()
        if self.tick_count >= self.max_tick_count:
            self.done = True

        return self.getObservation(), self.rewardCalculation(), self.done, self.info

    def saveActionList(self):
        filename = "actionList.txt"
        with open(filename, "a") as file:
            file.write(str(self.actionsList) + "\n")

    def getObservation(self):
        velW = self.walker.carlaWalker.get_velocity()
        velocityWalker = [velW.x, velW.y]

        velV = self.vehicle.vehicleCarla.get_velocity()
        velocityVehicle = [velV.x, velV.y]

        dirW = self.walker.carlaWalker.get_control().direction
        directionWalker = [dirW.x, dirW.y]

        posWaler = self.walker.getPositionXY()
        posCar = self.vehicle.getPositionXY()
        imitation_walker_xy = self.positions_walker[self.tick_count]
        obs = np.array(posCar + velocityVehicle + posWaler + velocityWalker + directionWalker + imitation_walker_xy)
        return obs

    def relativeAngle(self):
        posCar = self.vehicle.getPositionXY()
        posWalker = self.walker.getPositionXY()
        x_rel = posCar[0] - posWalker[0]
        y_rel = posCar[1] - posWalker[1]
        return self.walker.carlaWalker.get_transform().rotation.yaw - math.degrees(math.atan2(y_rel, x_rel))

    def rewardPoint_in_Polys(self, point_x_y, polys=None):

        point2 = self.positions_walker[self.tick_count]

        # if dist_xy(point_x_y, point2) <20:
        #     self.connection.draw_waypoint(carla.Location(x=point2[0],
        #         y=point2[1], z=8), "°", life_time=0.02, intensity=1)
        if polys is None:
            polys = []
            point1 = (159.5, -80)
            point2 = (170, -80)
            point3 = (170, -132)
            point4 = (159.5, -132)
            polys.append(Polygon([point1, point2, point3, point4]))
            point5 = (160, -124)
            point6 = (160, -120)
            point7 = (143, -121)
            point8 = (143, -125)
            polys.append(Polygon([point5, point6, point7, point8]))
        point = Point(point_x_y[0], point_x_y[1])
        dist_min = float('inf')
        for poly in polys:
            if poly.contains(point):
                return 0
            dist = poly.distance(point)
            if dist < dist_min:
                dist_min = dist
        # self.connection.draw_waypoint(carla.Location(x=point_x_y[0], y=point_x_y[1], z=8), "°", life_time=3,
        #                               intensity=1)
        return max(-dist_min / 2, -1)

    def rewardCalculation_imitation(self):
        point = self.positions_walker[self.tick_count]
        distance = math.dist(self.walker.getPositionXY(), point)
        if distance < 20:
            return min(0, (- distance) / 20)
        return -1

    # Calculates the reward based on distance, time-to-collision (TTC), and angle, as well as any additional bonuses
    # for specific conditions.
    def rewardCalculation(self):
        # == Continuos Rewards ==

        # Distance Reward between -1 and 0
        reward_distance = math.dist(self.walker.getPositionXY(), self.vehicle.getPositionXY())
        reward_distanceNormalize = - (reward_distance) / self.maxPunishmentDistance
        reward_distanceNormalize = max(-1, reward_distanceNormalize)
        reward_distanceNormalize = min(0, reward_distanceNormalize)
        # TTC Reward between -1 and 0
        ttc = self.ttc()
        if ttc < self.min_ttc:
            self.min_ttc = ttc

        self.lastTTC.append(ttc)
        if len(self.lastTTC) > 3:
            self.lastTTC.pop(0)
        ttcNormalize = - (max(min(self.lastTTC), -self.punishmentTTC) / self.punishmentTTC)

        # Angle in deg rel to walker yaw rotation Reward between -1 und 0
        relAngle = abs(self.relativeAngle())
        eyeSignDegree = 90
        if relAngle > eyeSignDegree:
            # Not in eye sight
            angleReward = 0
        else:
            # Punish small angle
            angleReward = - ((eyeSignDegree - relAngle) / eyeSignDegree)
        # reward_distanceNormalize (-1; 0)
        # ttcNormalize (-1; 0)
        # angleReward (-1; 0)

        norm_ttc_min = - self.min_ttc / 3
        norm_dist_walker = self.rewardCalculation_imitation()
        crosswalkReward = self.rewardPoint_in_Polys(self.walker.getPositionXY())

        self.distenceRewards.append(reward_distanceNormalize)
        self.ttcRewards.append(ttcNormalize)
        self.angleRewards.append(angleReward)
        self.minttcRewards.append(norm_ttc_min)
        self.walkerdistRewards.append(norm_dist_walker)
        self.crosswalkRewards.append(crosswalkReward)

        continiues_rewards = \
            self.rewardDistribution[0] * angleReward + \
            self.rewardDistribution[1] * ttcNormalize + \
            self.rewardDistribution[2] * reward_distanceNormalize + \
            self.rewardDistribution[3] * norm_ttc_min + \
            self.rewardDistribution[4] * norm_dist_walker + \
            crosswalkReward

        self.basicReward += \
            0.0 * angleReward + \
            0.0 * ttcNormalize + \
            0.0 * reward_distanceNormalize + \
            1.0 * norm_ttc_min + \
            0.0 * self.rewardCalculation_imitation() + \
            crosswalkReward
        # print(ttc, ttcNormalize, reward_distance, reward_distanceNormalize, relAngle, angleReward,
        # continues_rewards) == spare rewards ==
        # Fix Malfunction in collision event
        if self.collisionReward > 0 and self.tick_count > 20:
            self.done = True
        else:
            self.collisionReward = 0


        if self.done:
            self.basicReward += self.collisionReward
            self.rewardGesamt += continiues_rewards + self.collisionReward
            return continiues_rewards + self.collisionReward
        self.rewardGesamt += continiues_rewards
        return continiues_rewards

    # Resets the environment, resetting all relevant variables and objects.
    def reset(self):
        """ Reset all values """

        self.collisionReward = 0
        self.walker.resetWalker()
        self.vehicle.reset_car()
        if self.min_ttc_env > self.min_ttc:
            self.min_ttc_env = self.min_ttc
        self.lastBasicReward = self.basicReward
        self.lastmin_ttc = self.min_ttc
        self.min_ttc = 3
        if self.basicReward > self.highest_basic_reward and self.basicReward != 0 and self.done:
            if sum(self.lastTTC) < 9 or self.tick_count >= self.max_tick_count -1:
                self.highest_basic_reward = self.basicReward
                # if self.min_ttc_env < 0.5:
                #     print("Min TTC: ", self.min_ttc_env, ":", self.actionsList)
        self.done = False
        self.actionsList = []
        self.connection.world.tick()
        self.tick_count = 0
        self.basicReward = 0
        self.rewardGesamt = 0
        self.lastTTC = []
        self.angleRewards = []
        self.distenceRewards = []
        self.ttcRewards = []
        self.minttcRewards = []
        self.walkerdistRewards = []
        self.collisionRewards = []
        self.crosswalkRewards = []

        return self.getObservation()

    # Destroys the vehicle and walker objects and performs necessary cleanup.
    def close(self):
        # Destroy Car
        self.vehicle.vehicleCarla.set_autopilot(False)
        self.vehicle.collisionSensor.destroy()
        self.vehicle.vehicleCarla.destroy()

        # Destroy Walker
        self.walker.collisionSensor.destroy()
        self.walker.carlaWalker.destroy()

        # tick for changes to take effect
        self.connection.world.tick()
        self.connection.world.tick()
        self.connection.world.tick()

    # Calculates the time-to-collision (TTC) between the walker and vehicle.
    def ttc(self):
        posWalker = self.walker.getPositionXY()
        posVehicle = self.vehicle.getPositionXY()

        rot1 = math.radians(self.walker.carlaWalker.get_transform().rotation.yaw)
        rot2 = math.radians(self.vehicle.vehicleCarla.get_transform().rotation.yaw)

        vel1 = velocity_3d_to_ms(self.walker.carlaWalker.get_velocity())
        vel2 = velocity_3d_to_ms(self.vehicle.vehicleCarla.get_velocity())

        # New Pos for Walker, where center of Walker is in the back middle
        posWalker = [posWalker[0] - self.lengthWalker / 2 * math.cos(rot1),
                     posWalker[1] - self.lengthWalker / 2 * math.sin(rot1)]

        length = vel1 * self.timeLookAhead + self.lengthWalker
        length2 = vel2 * self.timeLookAhead + self.lengthVehicle
        poly1 = calc_bounding_box(p0=posWalker, angle=rot1, length=length, width=self.widthWalker)
        poly2 = calc_bounding_box(p0=posVehicle, angle=rot2, length=length2, width=self.widthVehicle)
        intersection = poly1.intersects(poly2)

        if intersection:
            # Check if collision in first halt
            length = vel1 * self.timeLookAhead / 2 + self.lengthWalker
            length2 = vel2 * self.timeLookAhead / 2 + self.lengthVehicle
            poly1 = calc_bounding_box(p0=posWalker, angle=rot1, length=length, width=self.widthWalker)
            poly2 = calc_bounding_box(p0=posVehicle, angle=rot2, length=length2, width=self.widthVehicle)
            if poly1.intersects(poly2):
                return self.ttc_collision_search(0, self.timeLookAhead / 2, vel1, vel2, rot1, rot2, posWalker,
                                                 posVehicle)
            else:
                return self.ttc_collision_search(self.timeLookAhead / 2, self.timeLookAhead, vel1, vel2, rot1, rot2,
                                                 posWalker,
                                                 posVehicle)
        return self.punishmentTTC

    def ttc_collision_search(self, timetoCollisionSearch, timeLookAhead, vel1, vel2, rot1, rot2, posWalker, posVehicle):
        while timetoCollisionSearch < timeLookAhead:
            length = vel1 * timetoCollisionSearch
            length2 = vel2 * timetoCollisionSearch
            x = posWalker[0] + length * math.cos(rot1)
            y = posWalker[1] + length * math.sin(rot1)
            x2 = posVehicle[0] + length2 * math.cos(rot2)
            y2 = posVehicle[1] + length2 * math.sin(rot2)

            polyWalker = calc_bounding_box(p0=[x, y], angle=rot1, length=self.lengthWalker, width=self.widthWalker)
            polyCar = calc_bounding_box(p0=[x2, y2], angle=rot2, length=self.lengthVehicle, width=self.widthVehicle)

            intersectionAhead = polyWalker.intersects(polyCar)

            if intersectionAhead:
                return timetoCollisionSearch

            if timetoCollisionSearch < 0.1:
                timetoCollisionSearch += 0.01
            else:
                timetoCollisionSearch += 0.1
        return self.punishmentTTC

    def showCarPoint(self):
        posVehicle = self.vehicle.getPositionXY()
        rot2 = math.radians(self.vehicle.vehicleCarla.get_transform().rotation.yaw)
        poly2 = calc_bounding_box(p0=posVehicle, angle=rot2, length=2.36, width=1)
        x, y = poly2.exterior.xy
        for i, j in zip(x, y):
            location = carla.Location(i, j, 4)
            self.connection.draw_waypoint(location=location, index='o', life_time=1, intensity=1)
