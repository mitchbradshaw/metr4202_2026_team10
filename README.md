# metr4202_2026_team10

**Summary**

This project implements exploration and target search for the TurtleBot3 Waffle Pi using ROS2 and Gazebo. The robot explores an unknown environment by finding frontier cells and navigating towards unexplored aeras using Nav2. While exploring, the robot uses its front-facing camera to detect ArUco markers. The environment is mapped using LiDAR-based SLAM.

The repository contains five packages:
- exploration: Detects frontiers, groups them as clusters and finds waypoint for best cluster.

- mapping_pkg: Camera processing and detects ArUco markers, then places them on the map.

- mapping_interfaces: ROS 2 interfaces (the marker detection message passed between the two camera nodes).

- team10_bringup: Launch files that start the whole system.

- metr4202_test_worlds: Different Gazebo worlds for testing.

Nav2, slam_toolbox and the TurtleBot3 packages are used as installed, with their default parameters. The other packages are our own nodes and launch files.


**Installation** (Ubuntu 22.04, ROS 2 Humble, Gazebo Classic 11)

1. ROS packages we build on
```
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox \
    ros-humble-turtlebot3 ros-humble-turtlebot3-simulations ros-humble-gazebo-ros-pkgs \
    ros-humble-cv-bridge ros-humble-tf2-geometry-msgs python3-opencv
```
2. Clone this repo and the course world package into the src folder of a workspace
```
mkdir -p ~/metr4202_ws/src && cd ~/metr4202_ws/src
git clone https://github.com/mitchbradshaw/metr4202_2026_team10.git
git clone https://github.com/METR4202/metr4202_aruco_explore.git
```
3. Build from the workspace root (not inside the repo)
```
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
cd ~/metr4202_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```
Every new terminal needs the two `source` and `export` lines again.


**Running in simulation**

Reference world, two terminals:
```
ros2 launch metr4202_aruco_explore metr4202_demo_world.launch.py
```
```
ros2 launch team10_bringup stack.launch.py
```
`stack.launch.py` starts Nav2 with SLAM, the camera nodes and the explorer. It does not start Gazebo, so it works with any world that is already running.

Alternatively, start SLAM and Nav2 in their own terminal and only our nodes on top :
```
ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True slam:=True
```
```
ros2 launch team10_bringup team10.launch.py use_sim_time:=true config:=sim
```
`team10.launch.py` starts only the camera nodes and the explorer.

Our own test worlds, world and stack together:
```
ros2 launch team10_bringup full_system.launch.py world:=marker_gauntlet
```
`world` can be `horseshoe`, `marker_gauntlet` or `slam_stress` (see `metr4202_test_worlds/README.md`).

What you should see: the robot spawns, Nav2 and RViz start after about 10 s, and after about 20 s the explorer prints `Frontier explorer started.` Then, for every waypoint it sends, it prints the five best candidates (position, gain, distance and utility, best first) followed by a `Chosen goal (x, y)` line. In RViz, add a MarkerArray display on `/aruco/map_markers` to see the markers found so far and an Image display on `/aruco/debug_image` for the camera view. A MarkerArray display on `/exploration/candidates` shows why the current goal was picked: frontier cells in orange, one sphere per candidate from blue (lowest utility) to red (highest), grey for skipped ones, labels on the best five, and a green sphere on the chosen goal. Marker positions in the map frame are also written to `/tmp/aruco_markers.csv` every 5 s. When no frontiers are left the explorer says so and stops sending goals.


**Running on the real robot**

Bring up the TurtleBot as normal and check the camera topic name with `ros2 topic list | grep camera` (it is normally compressed, and has to match `image_topic` in `team10_bringup/config/rig.yaml`).

If SLAM and Nav2 are already running (started with `use_sim_time:=False`), start only our nodes:
```
ros2 launch team10_bringup team10.launch.py
```
Its defaults (`use_sim_time:=false`, `config:=rig`) are set for the real robot.

To start SLAM and Nav2 together with our nodes instead:
```
ros2 launch team10_bringup stack.launch.py use_sim_time:=false config:=rig
```


**Tuning parameters** (`team10_bringup/config`)

Every tuning value for our three nodes is in `params.yaml`: utility weights `a` and `b`, gain radius, goal timeouts, the camera queue depth, marker range and outlier limits, and so on. Each is commented there. `sim.yaml` and `rig.yaml` are loaded on top of it with the few values that differ between Gazebo and the real robot (camera topic, compressed or not, marker size), chosen with `config:=sim` or `config:=rig` (the default is `sim` for `stack.launch.py` and `full_system.launch.py`, and `rig` for `team10.launch.py`). Copying `rig.yaml` to, say, `rig_slow.yaml` gives a new option `config:=rig_slow`.

With `colcon build --symlink-install`, edits to these files take effect on the next launch with no rebuild (the one exception is a newly added file, which needs one build to be installed). `frontier_explorer` and `map_markers_node` also pick up changes while running:
```
ros2 param set /frontier_explorer a 2.0       # used from the next goal it picks
ros2 param get /frontier_explorer a
```
Keep the decimal point on decimal values (`2.0`, not `2`), or ROS rejects the value as the wrong type. `ros2 param set` changes only the running node, so copy a value you like back into the YAML. `camera_node` and `map_frame` are read only at startup. The explorer logs every value it is using when it starts.


**Exploration strategy** (`exploration` package)

1. Frontier cells (`frontier_detection.py`): a free cell with at least one unknown 4-connected neighbour, i.e. the edge between mapped and unmapped space.
2. Clusters (`waypoint_finder.py`): frontier cells are grouped by a breadth-first search over 8-connected neighbours. A cluster is capped at about 1.75 m of cells so a long frontier gives several waypoints, and clusters under 5 cells are ignored as noise.
3. Waypoint: the cluster cell nearest the centroid, converted to map coordinates, then moved to the nearest cell of Nav2's global costmap with cost 80 or less so the goal sits outside the inflation ring and Nav2 will accept it.
4. Score (`waypoint_decider.py`): `U = a * (gain + bonus) - b * distance`. Gain is the unknown area the lidar could see from the waypoint (120 rays out to 2.5 m, each stopping at the first wall). Bonus is `marker_bonus` (in m², like gain) for a marker view pose and 0 otherwise. Distance is the driving distance over the free cells of `/map` (`path_distance.py`), so a goal behind a wall costs the drive round it, and one the robot can't reach at all is skipped. Waypoints within 0.5 m of the robot or of a blacklisted goal are skipped too. Local first: the best waypoint within `local_radius_m` (3 m) wins unless a further one offers `far_ratio` (1.5) times as much `a * (gain + bonus)`, so the robot finishes the area it is in before crossing the map. All of these values are in `team10_bringup/config/params.yaml`.
5. Send (`explorer_node.py`): the goal goes out as a `PoseStamped` on `/goal_pose`, as in the Prac 4 waypoint cycler, and the node waits for `NavigateRecovery` to report `IDLE` on `/behavior_tree_log` before choosing the next one.

Failure detection: a goal is blacklisted if Nav2 goes idle while the robot is still more than 0.5 m away, or if it is still active after 60 s. If every waypoint ends up blacklisted the list is cleared and scoring runs again, so a stale entry cannot stall the robot. When no frontier cells remain, exploration is complete.

Exploration runs until there is no reachable frontier, so everything the lidar can reach gets mapped. Collision-free motion comes from Nav2 with the default Waffle Pi parameters; our part is to only send goals that are outside the inflation layer.


**Marker detection** (`mapping_pkg`)

`camera_node` runs OpenCV's ArUco detector (`DICT_6X6_1000`, 100 mm markers) on each frame using the intrinsics from `/camera/camera_info`, and publishes each marker's position relative to the camera on `/aruco/detections`. `map_markers_node` uses tf2 to find where the camera was when the image was taken and transforms the detection into the map frame. Detections beyond 2 m are ignored, each marker's position is a running average of its detections, and after five sightings any detection more than 1 m from the average is dropped as an outlier. These limits and the marker size are set in `team10_bringup/config`.


**Libraries used**

Nav2 and slam_toolbox (through `turtlebot3_navigation2`), the TurtleBot3 packages, tf2, cv_bridge and OpenCV's aruco module. The goal and behaviour-tree-log pattern follows the Prac 4 `waypoint_cycler`.


**Useful commands**
```
ros2 run turtlebot3_teleop teleop_keyboard                          # drive by hand
ros2 run rqt_image_view rqt_image_view                              # camera view
ros2 run tf2_ros tf2_echo map base_link                             # is SLAM publishing
python3 -m pytest exploration/test/test_exploration_logic.py -v    # frontier logic unit tests
```
