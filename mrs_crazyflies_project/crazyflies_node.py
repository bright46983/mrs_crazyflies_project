import rclpy
from rclpy.node import Node
from functools import partial

from geometry_msgs.msg import Twist, PoseStamped, Pose, Point
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray
from .utils.OccupancyMap import OccupancyMap
from nav2_msgs.srv import LoadMap

from .agent import Agent
from crazyflie_interfaces.msg import LogDataGeneric
import time

import numpy as np

def wrap_angle(ang):
    return ang + (2.0 * np.pi * np.floor((np.pi - ang) / (2.0 * np.pi)))

class CrazyFliesNode(Node):

    def __init__(self, num_agents):
        super().__init__('crazyflies_node')

        self.num_agents = num_agents
        self.map = None
        self.map_recieved = False
        self.trajectory = Path()
        
        self.enable_visualization = True
        self.visualize_array = MarkerArray()
        self.visualize_acc_array = MarkerArray()


        # Agent related Variable
        self.agent_list = []
        self.neighbor_list = [] # list of list of agents [[agent1,agent2], [agent4],....]
        self.num_leaders = 1

    
        # create a list of publisher, subscriber for each fly
        self.vel_pub_list = []
        self.traj_pub_list = []
        self.connection_list = []
        for i in range(num_agents):
            # add fly to agent list
            self.agent_list.append(Agent(id = i+1))

            # get connection information from param server
            self.declare_parameter("cf_{}_connections".format(i+1),[])
            cf_connections = self.get_parameter("cf_{}_connections".format(i+1)).value
            self.connection_list.append(cf_connections)

            # create subscription for each fly
            self.create_subscription(PoseStamped, '/cf_{}/pose'.format(i+1), 
                                    partial(self.pose_cb,i),10)
            self.create_subscription(LogDataGeneric, '/cf_{}/velocity'.format(i+1), 
                                    partial(self.vel_cb,i),10)

            # create publisher for each fly
            vel_pub = self.create_publisher(Twist, '/cf_{}/cmd_vel'.format(i+1), 10)
            self.vel_pub_list.append(vel_pub) 
            vel_pub = self.create_publisher(Twist, '/cf_{}/cmd_vel'.format(i+1), 10)


        # others subscription
        self.create_subscription(OccupancyGrid, '/map',self.map_cb,1)
        self.create_subscription(PoseStamped, '/goal_pose',self.goal_cb,10)
        

        # Formation related Variable
        self.protocol = "Flocking" # "Rendezvous", "Rectangle", "Triangle", "Line"
        self.A_matrix = self.create_A_matrix(self.protocol,self.connection_list)
        self.formation = self.create_formation(self.protocol)


        time.sleep(3)
        self.get_logger().info("waiting for takng off...")
        for vel_pub in self.vel_pub_list:
            takeoff_vel = Twist()
            takeoff_vel.linear.x = 0.05
            vel_pub.publish(takeoff_vel)

        time.sleep(10)

        self.get_logger().info("finished takng off...")


        self.dt = 0.1  # seconds
        
        for i in range(num_agents):
            self.create_timer(self.dt, partial(self.main_loop,i))
        self.create_timer(self.dt, self.neighbor_loop)
        
    ###############################################
    ###### Callback
    ###############################################
    def pose_cb(self, id, msg):
        self.agent_list[id].position.x = msg.pose.position.x
        self.agent_list[id].position.y = msg.pose.position.y

    def vel_cb(self, id, msg):
        self.agent_list[id].velocity.x = msg.values[0]
        self.agent_list[id].velocity.y = msg.values[1]
        if msg.values[0] != 0.0 and msg.values[1] != 0.0: # preserve current heading 
                self.agent_list[id].heading = np.arctan2(self.agent_list[id].velocity.y, self.agent_list[id].velocity.x)


    def goal_cb(self, msg):
        for i in range(self.num_leaders):
            self.agent_list[i].goal = msg.pose.position

    def map_cb(self, gridmap):
        self.get_logger().info("Map recieved")
        self.map = OccupancyMap()
        env = np.array(gridmap.data).reshape(gridmap.info.height, gridmap.info.width).T
        # Set avoid obstacles - The steer to avoid behavior (IN THE DICTIONARY) requires the map, resolution, and origin
        self.map.set(data=env, 
                    resolution=gridmap.info.resolution, 
                    origin=[gridmap.info.origin.position.x, gridmap.info.origin.position.y])
        self.map_recieved = True
    ###############################################
    ###### Timer loop
    ###############################################

    def main_loop(self,idx):
        
        # update neighbor
        if not self.neighbor_list == []: 
            self.agent_list[idx].neighbor_agents = self.neighbor_list[idx]
        # update map for each agent
        # self.agent_list[idx].update_perception_field(self.map)

        # caculate acc
        nav_acc = self.agent_list[idx].navigation_acc()
        sep_acc = self.agent_list[idx].seperation_acc()
        coh_acc = self.agent_list[idx].cohesion_acc()
        align_acc = self.agent_list[idx].allignment_acc()
        # obs_acc = self.agent_list[idx].obstacle_acc()
        zero = Point()
        obs_acc = zero
        self.acc_list = [nav_acc,sep_acc,coh_acc,align_acc,zero]

        all_acc = self.agent_list[idx].combine_acc_priority(nav_acc,sep_acc,coh_acc,align_acc,obs_acc)    

        out_vel = self.agent_list[idx].cal_velocity(all_acc,self.dt)

        # publish        
        vel_msg = Twist()
        vel_msg.linear.x = out_vel.x
        vel_msg.linear.y = out_vel.y
        self.vel_pub_list[idx].publish(vel_msg)

    def neighbor_loop(self):
        neighbor_list = []
        A_matrix = self.A_matrix.copy()
        for i in range(len(self.agent_list)):
            neighbor = []
            others_agent = self.agent_list.copy()
            agent = others_agent.pop(i)
            for o_agent in others_agent:
                dis = np.linalg.norm(np.array([agent.position.x, agent.position.y])-np.array([o_agent.position.x, o_agent.position.y]))
                ang = abs(wrap_angle(agent.heading - np.arctan2( o_agent.position.y - agent.position.y, o_agent.position.x- agent.position.x)))
                if dis < agent.neighbor_range and ang < agent.neightbor_angle:
                    neighbor.append(o_agent)
            neighbor_list.append(neighbor)
            A_matrix[i][neighbor.id-1] = 1
        
        self.A_matrix = A_matrix
        self.neighbor_list = neighbor_list
            
    ###############################################
    ###### Other functions
    ###############################################
    def create_A_matrix(self,protocol,connections):
        A_matrix = np.zeros((self.num_agents,self.num_agents))
        if protocol == "Rendezvous": # fixed A matrix depend on config file
            for i in range(len(connections)): # each reciving node
                for c in connections[i]: 
                    A_matrix[i][c-1] = 1
            return A_matrix
        else: # other protocol will be later updated by neighbor
            return A_matrix
        
    # def update_A_from_neighbor(self,neighbors):
    #     A_matrix = np.zeros((self.num_agents,self.num_agents))
        
    #     for i in range(len(neighbors)): 
    #         for agent in neighbors[i]: 
    #             A_matrix[i][agent.id-1] = 1
    #     return A_matrix
    
    def create_formation(self,protocol):
        formation = np.zeros((self.num_agents, 2))
        if protocol == "Rectangle":
            if self.num_agents < 4:
                self.get_logger().error("Agent not enough for Rectangle formation")
                return list(formation)
            w = 1.0
            # fill out the corner 
            formation[0] = [0,0]
            formation[1] = [w,0]
            formation[2] = [w,w]
            formation[3] = [0,w]
            
            # put the rest in between
            #TO DO
            return list(formation)
        
        elif protocol == "Triangle":
            if self.num_agents < 3:
                self.get_logger().error("Agent not enough for Triangle formation")
                return list(formation)
            w = 1.0
            # fill out the corner 
            formation[0] = [0,0]
            formation[1] = [w,-w]
            formation[2] = [-w,-w]
            # put the rest in between
            # TODO
            formation[3] = [0,w] # menawhile just put in between
                
            return list(formation)
        elif protocol == "Line":
            # TODO
            return list(formation)
        else:
            return list(formation)
        
    ###############################################
    ###### Visualization
    ###############################################
    def visualize(self,_):
        if self.enable_visualization:
            self.visualize_goal()
            #self.visualize_acc()
            self.update_trajectory()
            
            # self.visual_pub.publish(self.visualize_array)
            # self.visual_pub_acc.publish(self.visualize_acc_array)

    def visualize_goal(self):
        if self.agent_list[0].goal:
            frame_id = "map"
            ns = "goal"

            marker = self.create_marker(666,ns, Marker.SPHERE, [self.agent_list[0].goal.x,self.agent_list[0].goal.y,0.2], 
                [0.3,0.3,0.3], [1,0,0,1], frame_id, None)
            self.visualize_array.markers.append(marker)

    def update_trajectory(self):
        # Create a new pose stamped with the current position
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "world"
        pose_stamped.pose.position = self.boid.position

        # Update the trajectory for the specified robot
        
        self.trajectory.poses.append(pose_stamped)

        traj_to_keep = 500
        if len(self.trajectory.poses) > traj_to_keep:
            self.trajectory.poses = self.trajectory.poses[-traj_to_keep:]

        self.trajectory.header.frame_id = "map"

        self.trajectory_pubs.publish(self.trajectory)
    
    def create_marker(self, marker_id, ns, marker_type, position, scale, color, frame_id,points):
        marker = Marker()
        marker.header.frame_id = frame_id  # Reference frame (change as necessary)
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD

        # Set marker position
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0


def main(args=None):
    rclpy.init(args=args)

    cfn = CrazyFliesNode(7)

    rclpy.spin(cfn)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    cfn.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()