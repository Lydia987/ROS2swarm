#!/usr/bin/env python3

import launch_ros.actions
from launch import LaunchDescription
from launch.actions import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    """Start the nodes required for the search object pattern."""

    robot_namespace = LaunchConfiguration('robot_namespace', default='robot_namespace_default')
    config_dir = LaunchConfiguration('config_dir', default='config_dir_default')
    log_level = LaunchConfiguration("log_level", default='debug')

    ld = LaunchDescription()
    ros2_pattern_node = launch_ros.actions.Node(
        package='ros2swarm',
        executable='search_object_pattern',
        namespace=robot_namespace,
        output='screen',
        parameters=[
            PathJoinSubstitution([config_dir, 'movement_pattern', 'basic', 'search_object_pattern.yaml'])],
        arguments=['--ros-args', '--log-level', log_level],
        on_exit=Shutdown()
    )
    ld.add_action(ros2_pattern_node)

    # ros2_pattern_subnode_random_walk = launch_ros.actions.Node(
    #     package='ros2swarm',
    #     executable='random_walk_pattern',
    #     namespace=robot_namespace,
    #     output='screen',
    #     remappings=[
    #         (['/', robot_namespace, '/drive_command'],
    #          ['/', robot_namespace, '/drive_command_random_walk_pattern'])
    #     ],
    #     parameters=[
    #         PathJoinSubstitution([config_dir, 'movement_pattern', 'basic', 'search_object_pattern.yaml'])],
    #     arguments=['--ros-args', '--log-level', log_level]
    # )
    # ld.add_action(ros2_pattern_subnode_random_walk)

    return ld
