# Represents a walker in the CARLA simulation. It contains methods for initializing the walker, applying control
# commands, and retrieving the walker's position.
import math

import carla
import numpy as np


class WalkerClass:
    # Class for the walker
    def __init__(self, world, spawnPoint_init, rotation):
        self.world = world
        self.spawnPoint_init = spawnPoint_init
        self.rotation_init = rotation
        self.max_walking_speed = 5
        self.carlaWalker, self.collisionSensor = self.__spawn_walker()
        # bounding_box = self.carlaWalker.bounding_box
        # print("bounding_box Walker: ", bounding_box.extent, "Zenter: ", bounding_box.location)


    def __spawn_walker(self):
        """ Load Blueprint and spawn walker """
        walker_bp = self.world.get_blueprint_library().filter('0012')[0]
        walker_spawn_transform = carla.Transform(location=self.spawnPoint_init, rotation=self.rotation_init)
        walker = self.world.spawn_actor(walker_bp, walker_spawn_transform)
        try:
            collision_sensor_walker = self.world.spawn_actor(
                self.world.get_blueprint_library().find('sensor.other.collision'),
                carla.Transform(), attach_to=walker)
        except:
            print("collision sensor failed")
        return walker, collision_sensor_walker

    # Get x and y position of walker
    def getPositionXY(self):
        return [self.carlaWalker.get_transform().location.x,
                self.carlaWalker.get_transform().location.y]

    # Apply control to walker
    def apply_control(self, action):
        walker_speed_action = action[2]
        action = np.array([action[0], action[1]])
        action_length = np.linalg.norm(action)
        if action_length == 0.0:
            # the chances are slim, but theoretically both actions could be 0.0
            unit_action = np.array([0.0, 0.0], dtype=np.float32)
        elif action_length > 1.0:
            # create a vector for the action with the length of zero
            unit_action = np.array(action / action_length)
        else:
            unit_action = np.array(action)
        direction = carla.Vector3D(x=float(unit_action[0]), y=float(unit_action[1]), z=0.0)
        walker_speed = walker_speed_action * self.max_walking_speed
        walker_control = carla.WalkerControl(
            direction, speed=walker_speed, jump=False)

        self.carlaWalker.apply_control(walker_control)

        unit_action = np.append(unit_action, [walker_speed])
        return unit_action.tolist()

    # Reset walker to initial position
    def resetWalker(self):
        try:
            self.collisionSensor.destroy()
            self.carlaWalker.destroy()
            self.carlaWalker, self.collisionSensor = self.__spawn_walker()
        except:
            print("Fail to destroy Walker")


# Represents a vehicle in the CARLA simulation. It contains methods for initializing the vehicle, resetting its
# position, and interacting with the CustomEnv environment.
class VehicleClass:
    # Class for the vehicle
    def __init__(self, connection, spawnPoint_init=carla.Location(x=-39.944832, y=-3.153754, z=2.150400),
                 force3D=carla.Vector3D(x=0, y=0, z=0), rotation=carla.Rotation(pitch=0, yaw=180, roll=0),
                 environment=None):
        self.client = connection.client
        self.world = connection.world
        self.set_tm_seed()
        self.force3D = force3D
        self.spawnPoint_location = spawnPoint_init
        self.spawnPoint_rotation = rotation
        self.vehicleCarla, self.collisionSensor = self.__spawn_car(self.force3D)

        # Abmessungen des Tesla Model 3-Fahrzeugs abrufen
        # bounding_box = self.vehicleCarla.bounding_box
        # print("bounding_box Vehicle: ", bounding_box.extent, "Zenter: ", bounding_box.location)

        self.tick_count = 0
        self.env = environment
        # connection.draw_waypoint(spawnPoint_init, "o")

    # Spawn vehicle
    def __spawn_car(self, addForce3D):
        """ Spawn Car on a given position and activate autopilot"""
        tm_port = self.set_tm_seed()
        car_bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]

        car_sp = carla.Transform(location=self.spawnPoint_location, rotation=self.spawnPoint_rotation)
        car = self.world.spawn_actor(car_bp, car_sp)
        # Force to start with 8m/s
        velocity = addForce3D
        car.set_target_velocity(velocity)
        car.set_autopilot(True, tm_port)

        collision_sensor_car = self.world.spawn_actor(
            self.world.get_blueprint_library().find('sensor.other.collision'),
            carla.Transform(), attach_to=car)
        collision_sensor_car.listen(lambda event: self.collision_handler(event))
        return car, collision_sensor_car

    # Set seed for traffic manager
    def set_tm_seed(self):
        """ Set the seed of traffic manger"""
        # === Set Seed for TrafficManager ===
        seed_value = 0
        tm = self.client.get_trafficmanager(8000)
        tm_port = tm.get_port()
        tm.set_random_device_seed(seed_value)
        return tm_port

    # Get x and y position of vehicle
    def getPositionXY(self):
        return [self.vehicleCarla.get_transform().location.x,
                self.vehicleCarla.get_transform().location.y]

    # Collision handler in case vehicle collide with walker
    def collision_handler(self, event):
        """ handle collisions and calculate extra reward """
        actor_we_collide_against = event.other_actor
        impulse = event.normal_impulse # Vector3D N*s
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        # To ensure that the initial force does not count as collision
        collisionReward = 0
        if self.env.tick_count > 30:
            collisionReward = min(abs(intensity) * 100, 100)
        else:
            collisionReward = 0

        if (actor_we_collide_against.type_id == "walker.pedestrian.0012"):
            v_car = self.vehicleCarla.get_velocity()
            v_ms = math.sqrt(v_car.x * v_car.x + v_car.y * v_car.y + v_car.z * v_car.z)

            if v_ms > 1:
                # self.env.done = True
                self.env.collisionReward = collisionReward + v_ms / 10
                if collisionReward > 0.01:
                    print(f"Car Collition with Pedestrian: {collisionReward + v_ms / 10}")
                    # print(f"Actions: {self.env.actionsList}")
        else:
            ...
    # Reset vehicle to initial position
    def reset_car(self):
        tm_port = self.set_tm_seed()
        self.vehicleCarla.set_autopilot(False, tm_port)
        self.collisionSensor.destroy()
        self.vehicleCarla.destroy()
        self.vehicleCarla, self.collisionSensor = self.__spawn_car(self.force3D)
        self.tick_count = 0