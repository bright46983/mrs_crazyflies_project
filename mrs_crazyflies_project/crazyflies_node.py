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
        self.trajectory = []
        self.neighbor_list = [] # list of list of agents [[agent1,agent2], [agent4],....]
        self.num_leaders = 1

    
        # create a list of publisher, subscriber for each fly
        self.vel_pub_list = []
        self.traj_pub_list = []
        self.connection_list = []
        for i in range(num_agents):
            # add fly to agent list
            self.agent_list.append(Agent(id = i+1))
            self.trajectory.append(Path())

            # get connection information from param server
            self.declare_parameter("cf_{}_connections".format(i+1),[0])
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
            traj_pub = self.create_publisher(Path, '/cf_{}/path'.format(i+1), 10)
            self.traj_pub_list.append(traj_pub) 


        # others subscription
        self.create_subscription(OccupancyGrid, '/map',self.map_cb,1)
        self.create_subscription(PoseStamped, '/goal_pose',self.goal_cb,10)
        self.visual_pub = self.create_publisher(MarkerArray,'/cf/visualize',10)
        

        # Formation related Variable
        self.protocol = "Rendezvous" # "Flocking","Rendezvous", "Rectangle", "Triangle", "Line"
        self.A_matrix = self.create_A_matrix(self.protocol,self.connection_list)
        self.formation = self.create_formation(self.protocol)


        time.sleep(3)
        self.get_logger().info("waiting for takng off...")
        for vel_pub in self.vel_pub_list:
            takeoff_vel = Twist()
            takeoff_vel.linear.x = 0.05
            vel_pub.publish(takeoff_vel)

        # time.sleep(10)

        self.get_logger().info("finished takng off...")


        self.dt = 0.1  # seconds
        
        for i in range(num_agents):
            self.create_timer(self.dt, partial(self.main_loop,i))
        self.create_timer(self.dt, self.neighbor_loop)
        self.create_timer(0.5, self.visualize)

        
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
        if not self.neighbor_list == [] and self.protocol != "Rendezvous": 
            self.agent_list[idx].neighbor_agents = self.neighbor_list[idx]
        # update map for each agent
        # self.agent_list[idx].update_perception_field(self.map)

        # Reynold 
        zero = Point()
        nav_acc = zero
        sep_acc = self.agent_list[idx].seperation_acc()
        coh_acc = zero
        align_acc = zero
        obs_acc = zero
        # obs_acc = self.agent_list[idx].obstacle_acc()

        if self.protocol == "Flocking" :
            nav_acc = self.agent_list[idx].navigation_acc()
            coh_acc = self.agent_list[idx].cohesion_acc()
            align_acc = self.agent_list[idx].allignment_acc()
        
        self.acc_list = [nav_acc,sep_acc,coh_acc,align_acc,obs_acc]

        all_acc = self.agent_list[idx].combine_acc_priority(nav_acc,sep_acc,coh_acc,align_acc,obs_acc)    

        reynold_vel = self.agent_list[idx].cal_velocity(all_acc,self.dt)

        # Consensus
        consensus_vel = zero
        if self.protocol == "Rendezvous":
            consensus_vel = self.cal_rendezvous_vel(self.agent_list,self.A_matrix)
        elif self.protocol != "Flocking":
            consensus_vel = self.cal_formation_vel(self.agent_list,self.A_matrix,self.formation)
            
        # combine velocity (Reynold + Consensus)
        out_vel = self.combine_vel(reynold_vel ,consensus_vel)
        # publish        
        vel_msg = Twist()
        vel_msg.linear.x = out_vel.x
        vel_msg.linear.y = out_vel.y
        self.vel_pub_list[idx].publish(vel_msg)

    def neighbor_loop(self):
        if self.protocol != "Rendezvous":
            neighbor_list = []
            A_matrix = np.zeros((self.num_agents,self.num_agents))
            for i in range(len(self.agent_list)):
                neighbor = []
                others_agent = self.agent_list.copy()
                agent = others_agent.pop(i)
                for o_agent in others_agent:
                    dis = np.linalg.norm(np.array([agent.position.x, agent.position.y])-np.array([o_agent.position.x, o_agent.position.y]))
                    ang = abs(wrap_angle(agent.heading - np.arctan2( o_agent.position.y - agent.position.y, o_agent.position.x- agent.position.x)))
                    if dis < agent.neighbor_range and ang < agent.neightbor_angle:
                        neighbor.append(o_agent)
                        A_matrix[i][o_agent.id-1] = 1
                neighbor_list.append(neighbor)
                
            
            self.A_matrix = A_matrix
            self.neighbor_list = neighbor_list

    ###############################################
    ###### Consensus
    ###############################################
    def cal_rendezvous_vel(self,agents_list, A):
        out_vel = Point()
        # TODO

        return out_vel 

    def cal_formation_vel(self,agents_list, A, formation):
        out_vel = Point()
        # TODO
        return out_vel 
    
    def combine_vel(self,reynold,consensus):
        out_vel = Point()
        out_vel.x = reynold.x + consensus.x
        out_vel.y = reynold.y + consensus.y
        return out_vel
            
    ###############################################
    ###### Other functions
    ###############################################
    def create_A_matrix(self,protocol,connections):
        A_matrix = np.zeros((self.num_agents,self.num_agents))
        if protocol == "Rendezvous": # fixed A matrix depend on config file
            for i in range(len(connections)): # each reciving node
                for c in connections[i]:
                    if c != 0: 
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
            formation[3] = [-w,-w]
            # put the rest in between
            # TODO
            formation[2] = [0,-w] # menawhile just put in between
                
            return list(formation)
        elif protocol == "Line":
            # TODO
            return list(formation)
        else:
            return list(formation)
        
    ###############################################
    ###### Visualization
    ###############################################
    def visualize(self):
        if self.enable_visualization:
            self.visualize_goal()
            #self.visualize_acc()
            self.update_trajectory()
            self.visualize_connections()
            if self.protocol != "Flocking" and self.protocol != "Rendezvous":
                self.visualize_formation()
            print(self.neighbor_list)
            print(self.A_matrix)
            
            self.visual_pub.publish(self.visualize_array)
            self.visualize_array = MarkerArray()
            # self.visual_pub_acc.publish(self.visualize_acc_array)

    def visualize_goal(self):
        if self.agent_list[0].goal:
            frame_id = "world"
            ns = "goal"

            marker = self.create_marker(666,ns, Marker.SPHERE, [self.agent_list[0].goal.x,self.agent_list[0].goal.y,0.2], 
                [0.3,0.3,0.3], [1.0,0.0,0.0,1.0], frame_id, None)
            self.visualize_array.markers.append(marker)

    def update_trajectory(self):
        for i in range(self.num_agents):
            # Create a new pose stamped with the current position
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = "world"
            pose_stamped.pose.position.x = self.agent_list[i].position.x
            pose_stamped.pose.position.y = self.agent_list[i].position.y

            # Update the trajectory for the specified robot
            
            self.trajectory[i].poses.append(pose_stamped)

            traj_to_keep = 500
            if len(self.trajectory[i].poses) > traj_to_keep:
                self.trajectory[i].poses = self.trajectory[i].poses[-traj_to_keep:]

            self.trajectory[i].header.frame_id = "world"

            self.traj_pub_list[i].publish(self.trajectory[i])

    def visualize_connections(self):
        frame_id = "world"
        ns = "connections"

        points = []
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if self.A_matrix[i][j] == 1:
                    points.append([self.agent_list[i].position.x ,self.agent_list[i].position.y ,0.0])
                    points.append([self.agent_list[j].position.x ,self.agent_list[j].position.y,0.0])

        marker = self.create_marker(222,ns, Marker.LINE_LIST, [0.0,0.0,0.0], 
                [0.04,0.04,0.04], [0.0,0.0,1.0,1.0], frame_id, points)
        self.visualize_array.markers.append(marker)
    
    def visualize_formation(self):
        frame_id = "world"
        ns = "formation"

        points = []
        for i in range(self.num_agents-1):
            points.append([self.formation[i][0] + self.agent_list[0].position.x ,self.formation[i][1] +self.agent_list[0].position.y ,0.0])
            points.append([self.formation[i+1][0] + self.agent_list[0].position.x ,self.formation[i+1][1] +self.agent_list[0].position.y ,0.0])
        points.append([self.formation[-1][0] + self.agent_list[0].position.x ,self.formation[-1][1] +self.agent_list[0].position.y ,0.0])
        points.append([self.formation[0][0] + self.agent_list[0].position.x ,self.formation[0][1] +self.agent_list[0].position.y ,0.0])
        print(points)
        marker = self.create_marker(787,ns, Marker.LINE_LIST, [0.0,0.0,0.0], 
                [0.04,0.04,0.04], [0.0,1.0,0.0,1.0], frame_id, points)
        self.visualize_array.markers.append(marker)
        
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

         # Set maker Points
        if marker_type == Marker.LINE_LIST:
            for point in points:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                marker.points.append(p)

        # Set marker scale
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]

        # Set marker color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]  # Alpha (transparency)

        return marker

        


def main(args=None):
    rclpy.init(args=args)

    cfn = CrazyFliesNode(4)

    rclpy.spin(cfn)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    cfn.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()