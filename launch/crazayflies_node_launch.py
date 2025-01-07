import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # load crazyflies
    connection_yaml = os.path.join(
        get_package_share_directory('mrs_crazyflies_project'),
        'config',
        'connections.yaml')

    with open(connection_yaml, 'r') as ymlfile:
        connections = yaml.safe_load(ymlfile)

    server_params = [connections]
    # for key, value in connections.items():
    #     if isinstance(value, list):
    #         server_params.append({key: value})
    # server_params = connections
    print(server_params)

    launch_description = []
    launch_description.append(
        Node(
            package='mrs_crazyflies_project',
            executable='crazyflies_node',
            name='crazyflies_node',
            output='screen',
            parameters=server_params
        ))
    
    
    # launch_description.append(       
    #     Node(
    #         package='rviz2',
    #         namespace='',
    #         executable='rviz2',
    #         name='rviz2',
    #         arguments=['-d' + os.path.join(get_package_share_directory('mrs_crazyflies'), 'config', 'config.rviz')],
    #         parameters=[{
    #             "use_sim_time": True,
    #         }]
    #     )
        
    # )
    return LaunchDescription(launch_description)
