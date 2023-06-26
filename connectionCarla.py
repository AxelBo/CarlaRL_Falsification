#!/usr/bin/env python

import glob
import logging
import os
import random
import sys

import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import time


class CarlaConnection():
    # Class for Carla Connection and basic settings
    def __int__(self, townName="Town03", host="localhost", reloadWorld=True,
                camaraLocation=carla.Location(x=-200, y=0, z=150), camRotation=None, render=True, syncMode=False, port=2000, delta=0.05, setCamerView=True):
        # === Carla ===
        self.host = host
        self.town = townName
        self.port = port
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()

        self.map = self.world.get_map()
        if not self.map.name.endswith(self.town):
            self.world = self.client.load_world(self.town)
            while not self.world.get_map().name.endswith(self.town):
                time.sleep(0.2)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
        time.sleep(2)
        if reloadWorld:
            self.client.reload_world(False)
        if syncMode:
            self.setSynchronous_mode(no_render=not render, reloadWorld=reloadWorld, delta=delta),
        if setCamerView:
            self.set_camara_view(camaraLocation, camRotation)

    # Reinitialize the connection data
    def reinitialize(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        time.sleep(1)

    # Set the Synchronous Mode and Rendering
    def setSynchronous_mode(self, delta=0.05, no_render=True, reloadWorld=True):
        settings = self.world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = delta
        settings.no_rendering_mode = no_render
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 5
        self.world.apply_settings(settings)
        if reloadWorld:
            self.client.reload_world(False)

    # Set the Camara View
    def set_camara_view(self, location, camRotation):
        # === Walker View Camera ===
        # X und Y sind gedreht so....
        spectator = self.world.get_spectator()
        location = location
        transform = carla.Transform(location, self.world.get_map().get_spawn_points()[0].rotation)
        if not camRotation:
            camRotation = carla.Rotation(pitch=-60)
        spectator.set_transform(carla.Transform(transform.location, camRotation))

    # Print position on the carla map
    def draw_waypoint(self, location, index, life_time=120.0, intensity=1):
        if intensity < 0.33:
            color = carla.Color(r=0, g=255, b=0)
        elif intensity < 0.66:
            color = carla.Color(r=250, g=250, b=0)
        else:
            color = carla.Color(r=250, g=0, b=0)
        self.world.debug.draw_string(location, str(index), draw_shadow=False,
                                     color=color, life_time=life_time,
                                     persistent_lines=True)
    # Get a list of all vehicles
    def getVehicleList(self):
        actors = self.world.get_actors().filter('vehicle.*')
        return [actor for actor in actors]

    # Get a list of all walkers
    def getWalkerList(self):
        actors = self.world.get_actors().filter('walker.*')
        return [actor for actor in actors]