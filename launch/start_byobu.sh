#!/bin/bash

# Kill any existing Byobu sessions
byobu kill-server

# Start a new Byobu session with the name "crazy_sim"
byobu new-session -d -s crazy_sim

# Open the first window and run the command for sitl_multiagent_text.sh
byobu rename-window -t crazy_sim:0 'sitl_multiagent'
byobu send-keys -t crazy_sim:0 'bash ~/CrazySim/ros2_ws/src/mrs_crazyflies/launch/sitl_multiagent_text.sh -m crazyflie -f $SPAWN_POSE_DOC -w $ENV_NAME' C-m

# Create the second window for waitForCfGazebo with sleep
byobu new-window -t crazy_sim:1 -n 'velmux'
byobu send-keys -t crazy_sim:1 'waitForCfGazebo; sleep 12; ros2 launch mrs_crazyflies cf_velmux_launch.py' C-m

# Create the third window for map_server_launch.py
byobu new-window -t crazy_sim:2 -n 'map_server'
byobu send-keys -t crazy_sim:2 'waitForCfGazebo; sleep 15; ros2 launch mrs_crazyflies map_server_launch.py map_yaml:=/home/tanakrit/CrazySim/ros2_ws/src/mrs_crazyflies/maps/$ENV_NAME/$ENV_NAME.yaml' C-m

# Create the fourth window for idle or additional commands (left blank as requested)
byobu new-window -t crazy_sim:3 -n 'teleop'
byobu send-keys -t crazy_sim:3 'waitForCfGazebo; sleep 16;ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/cf_1/cmd_vel' C-m

# Attach to the session
byobu attach-session -t crazy_sim
