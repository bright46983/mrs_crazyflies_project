import rclpy
from rclpy.node import Node
from functools import partial

from geometry_msgs.msg import Twist, PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid, Path, MapMetaData
from visualization_msgs.msg import Marker, MarkerArray
from .utils.OccupancyMap import OccupancyMap
from nav2_msgs.srv import LoadMap
from std_srvs.srv import Trigger

import yaml
import cv2
from std_msgs.msg import Header

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
        self.consensus_vel_list = []

        # create a list of publisher, subscriber for each fly
        self.vel_pub_list = []
        self.traj_pub_list = []
        self.connection_list = []
        
        for i in range(num_agents):
            # add fly to agent list
            self.agent_list.append(Agent(id = i+1))
            self.trajectory.append(Path())
            self.consensus_vel_list.append(Point())

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
        self.create_subscription(PoseStamped, '/goal_pose',self.goal_cb,10)
        self.visual_pub = self.create_publisher(MarkerArray,'/cf/visualize',10)
        
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map_field', 10)

        # Formation related Variable
        self.protocol = "Line" #"Rectangle" # "Flocking","Rendezvous", "Rectangle", "Triangle", "Line", "Pentagon", "Hexagon"
        self.use_fixed_connection = True
        self.A_matrix = self.create_A_matrix(self.protocol,self.connection_list)
        self.formation = self.create_formation(self.protocol)

        map_yaml_path = '/root/ros2_ws/src/mrs_crazyflies_project/resource/maps/simple_maze_10x10/simple_maze_10x10.yaml'
        map_image_path = '/root/ros2_ws/src/mrs_crazyflies_project/resource/maps/simple_maze_10x10/simple_maze_10x10.bmp'
        occupancy_map = self.load_map(map_yaml_path, map_image_path)

        self.map = OccupancyMap()
        env = np.array(occupancy_map.data).reshape(occupancy_map.info.height, occupancy_map.info.width).T

        # Set avoid obstacles - The steer to avoid behavior (IN THE DICTIONARY) requires the map, resolution, and origin
        self.map.set(data=env, 
                    resolution=occupancy_map.info.resolution, 
                    origin=[occupancy_map.info.origin.position.x, occupancy_map.info.origin.position.y])
        

        self.perception_field_publisher = self.create_publisher(OccupancyGrid, '/perception_field', 10)

        # formation service
        # self.seq = ["Triangle","Line","Line2"]
        # self.seq_idx = 0
        # self.form_srv = self.create_service(Trigger, 'change_formation', self.change_formation)

        time.sleep(3)
        self.get_logger().info("waiting for takng off...")
        for vel_pub in self.vel_pub_list:
            takeoff_vel = Twist()
            takeoff_vel.linear.x = 0.0
            vel_pub.publish(takeoff_vel)


        time.sleep(5)

        self.get_logger().info("finished takng off...")

        self.dt = 0.05  # seconds
    
        self.create_timer(self.dt, self.consensus_loop)
        for i in range(num_agents):
            self.create_timer(self.dt, partial(self.main_loop,i))
        
        self.create_timer(self.dt, self.neighbor_loop)
        self.create_timer(0.5, self.visualize)

    def publish_perception_field(self, perception_field:OccupancyMap):
        # Create and populate the OccupancyGrid message
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = "map"  # Make sure this matches your TF frames
        grid.info = MapMetaData()
        grid.info.resolution = perception_field.resolution
        grid.info.width = perception_field.map_dim[1]
        grid.info.height = perception_field.map_dim[0]

        # Set origin
        pose = Pose()
        pose.position.x = float(perception_field.origin[0])
        pose.position.y = float(perception_field.origin[1])
        pose.position.z = 0.0
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        grid.info.origin = pose

        map_transposed = perception_field.map.T
        map_transposed = map_transposed.flatten()
        grid.data = map_transposed.astype(int).tolist()

        # Publish the map
        self.perception_field_publisher.publish(grid)
    
    ###############################################
    ###### Callback
    ###############################################
    def pose_cb(self, id, msg):
        self.agent_list[id].position.x = msg.pose.position.x
        self.agent_list[id].position.y = msg.pose.position.y
        self.agent_list[id].position.z = msg.pose.position.z


    def vel_cb(self, id, msg):
        self.agent_list[id].velocity.x = msg.values[0]
        self.agent_list[id].velocity.y = msg.values[1]
        if msg.values[0] != 0.0 and msg.values[1] != 0.0: # preserve current heading 
                self.agent_list[id].heading = np.arctan2(self.agent_list[id].velocity.y, self.agent_list[id].velocity.x)


    def goal_cb(self, msg):
        for i in range(self.num_leaders):
            self.agent_list[i].goal = msg.pose.position

    # def change_formation(self,req,res):
        
    #     self.formation = self.create_formation(self.seq[self.seq_idx])
    #     self.seq_idx = (self.seq_idx + 1) % 3
    #     return res

    ###############################################
    ###### Timer loop
    ###############################################

    def main_loop(self,idx):
        # self.get_logger().info(f"Debug! agent number: {idx:.1f}")
        # update neighbor
        if not self.neighbor_list == []: 
            self.agent_list[idx].neighbor_agents = self.neighbor_list[idx]
        # update map for each agent
        # self.agent_list[idx].update_perception_field(self.map)

        # Reynold 
        zero = Point()
        # if idx == 0:
        nav_acc = self.agent_list[idx].navigation_acc()
        # else:
            # nav_acc = zero

        sep_acc = self.agent_list[idx].seperation_acc()
        coh_acc = zero
        align_acc = zero
        obs_acc = zero#self.agent_list[idx].obstacle_acc()

        if self.protocol == "Flocking" :
            coh_acc = self.agent_list[idx].cohesion_acc()
            align_acc = self.agent_list[idx].allignment_acc()
            
        self.acc_list = [nav_acc,sep_acc,coh_acc,align_acc,obs_acc]

        all_acc = self.agent_list[idx].combine_acc_priority(nav_acc,sep_acc,coh_acc,align_acc,obs_acc)    

        # reynold_vel = self.agent_list[idx].cal_velocity(all_acc,self.dt,obs_acc)
        reynold_vel = self.agent_list[idx].cal_velocity(all_acc,self.dt)

        # Consensus
        consensus_vel = zero
        if self.protocol != "Flocking":
            consensus_vel = self.consensus_vel_list[idx]

        #stubborn agent
        if  self.agent_list[idx].goal != None:
            consensus_vel = zero
            
        # combine velocity (Reynold + Consensus)
        # if idx == 2:
        #     self.get_logger().info(f"Debug! reynold_vel X: {reynold_vel.x:.3f}, consensus_vel X: {consensus_vel.x:.3f}, reynold_vel Y: {reynold_vel.y:.3f}, consensus_vel Y: {consensus_vel.y:.3f}")

        out_vel = self.combine_vel(reynold_vel ,consensus_vel)
        # publish        
        vel_msg = Twist()
        vel_msg.linear.x = out_vel.x
        vel_msg.linear.y = out_vel.y
        self.vel_pub_list[idx].publish(vel_msg)

    def neighbor_loop(self):
        neighbor_list = []
        A_matrix = np.zeros((self.num_agents,self.num_agents))
        for i in range(len(self.agent_list)):
            neighbor = []
            others_agent = self.agent_list.copy()
            agent = others_agent.pop(i)
            for o_agent in others_agent:
                dis = np.linalg.norm(np.array([agent.position.x, agent.position.y])-np.array([o_agent.position.x, o_agent.position.y]))
                ang = abs(wrap_angle(agent.heading - np.arctan2( o_agent.position.y - agent.position.y, o_agent.position.x- agent.position.x)))
                if dis <= agent.neighbor_range and ang <= agent.neightbor_angle:
                    neighbor.append(o_agent)
                    A_matrix[i][o_agent.id-1] = 1
            neighbor_list.append(neighbor)
            
        
        self.neighbor_list = neighbor_list
        if self.protocol != "Rendezvous" and not self.use_fixed_connection: 
            A_matrix[0,:] = 0
            self.A_matrix = A_matrix
            
            # stubborn agent
            # self.A_matrix[0,:] = 0

    def consensus_loop(self):
        if self.protocol == "Rendezvous":
            self.consensus_vel_list = self.cal_rendezvous_vel(self.agent_list,self.A_matrix)
        elif self.protocol != "Flocking":
            self.consensus_vel_list = self.cal_formation_vel(self.agent_list,self.A_matrix,self.formation)

    ###############################################
    ###### Consensus
    ###############################################
    def laplacian(self,A):
        # diagonal of laplacian = row sum of A
        L_diag = np.sum(A,axis=1)
        L = np.diag(L_diag)
        # fill in the rest of the elements of L
        L -= A
        L = L/(self.num_agents*4)
        return L
    
    def cal_rendezvous_vel(self,agents_list, A):
        out_vel_list = []

        L = self.laplacian(A) # convert A to L
        x = [[agent.position.x,agent.position.y] for agent in agents_list]
        x = np.array(x) # convert to np array

        # consensus equation
        x_dot = -L@x

        # convert to list of Point objects
        for i in range(x_dot.shape[0]):
            out_vel = Point()
            out_vel.x = x_dot[i,0]
            out_vel.y = x_dot[i,1]
            out_vel_list.append(out_vel)

        return out_vel_list

    def cal_formation_vel(self,agents_list, A, formation):
        # make sure the number of elements in formation is the same as the number of agents
        assert len(agents_list) == len(formation)

        out_vel_list = []

        L = self.laplacian(A) # convert A to L

        num_agents = len(agents_list)
        x = [[agents_list[i].position.x - formation[i][0],agents_list[i].position.y - formation[i][1]] for i in range(num_agents)]
        x = np.array(x) # convert to np array

        # consensus equation
        x_dot = -L@x

        # TODO: incorporate pinning control

        # convert to list of Point objects
        for i in range(x_dot.shape[0]):
            out_vel = Point()
            out_vel.x = x_dot[i,0]
            out_vel.y = x_dot[i,1]
            out_vel_list.append(out_vel)
        return out_vel_list
    
    def combine_vel(self,reynold,consensus):
        out_vel = Point()
        out_vel.x = reynold.x + consensus.x
        out_vel.y = reynold.y + consensus.y

        return self.agent_list[0].limit_vel(out_vel)
            
    ###############################################
    ###### Other functions
    ###############################################
    def create_A_matrix(self,protocol,connections):
        A_matrix = np.zeros((self.num_agents,self.num_agents))
        # if protocol == "Rendezvous": # fixed A matrix depend on config file
        for i in range(len(connections)): # each reciving node
            for c in connections[i]:
                if c != 0: 
                    A_matrix[i][c-1] = 1

        return A_matrix
        # else: # other protocol will be later updated by neighbor
        #     return A_matrix
        
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
            w = 1.5
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
            w = 2.0
            # fill out the corner 
            formation[0] = [0.0,0.0]
            # formation[1] = [w,-w]
            # formation[2] = [-w,-w]
            formation[1] = [0.4,0.8*0.866]
            formation[2] = [-0.4,0.8*0.866]
            # put the rest in between
            # TODO
            # formation[2] = [0,-w] # menawhile just put in between
                
            return list(formation)
        elif protocol == "Line":
            if self.num_agents < 2:
                self.get_logger().error("Agent not enough for Line formation")
                return list(formation)
            l = 0
            for i in range(self.num_agents):
                formation[i] = [0,l]
                l = l - 0.8
            # put the rest in between
            # TODO
            return list(formation)
        elif protocol == "Pentagon":
            if self.num_agents < 5:
                self.get_logger().error("Agent not enough for Pentagon formation")
                return list(formation)
            formation[0] = [0.0, 0.0]                    # First vertex (corner) at (0,0)
            formation[1] = [-0.5878,0.8090]              # Second vertex
            formation[2] = [-1.5388,0.5000]               # Third vertex
            formation[3] = [-1.5388,-0.5000]              # Fourth vertex
            formation[4] = [-0.5878,-0.8090]            # Fifth formation[]
            return list(formation)
        
        elif protocol == "Hexagon":
            if self.num_agents < 6:
                self.get_logger().error("Agent not enough for Hexagon formation")
                return list(formation)
            formation[0] = [0.0, 0.0]                  # First formation[] (corner) at (0,0)
            formation[1] = [0.8666,-0.5000]              # Second formation[]
            formation[2] = [0.8666,-1.5000]             # Third formation[]
            formation[3] = [0.0000,-2.0000]              # Fourth formation[]
            formation[4] = [-0.8666,-1.5000]            # Fifth formation[]
            formation[5] = [-0.8666,-0.5000]             # Sixth vertex
            return list(formation)     
        elif protocol == "Circle":
            if self.num_agents < 8:
                self.get_logger().error("Agent not enough for circle formation")
                return list(formation)
            for i in range(8):  # A hexagon has 10 vertices
                angle = np.pi/2 + 2 * np.pi * i / 8  # Divide the circle into 6 equal parts
                r = 1.2
                x = r * np.cos(angle)
                y = r - r * np.sin(angle)
                formation[i] = [x, y]
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
            # self.visualize_connections()
            if self.protocol != "Flocking" and self.protocol != "Rendezvous":
                self.visualize_formation()
            
            self.visual_pub.publish(self.visualize_array)
            self.visualize_array = MarkerArray()
            # self.visual_pub_acc.publish(self.visualize_acc_array)

    def visualize_goal(self):
        if self.agent_list[0].goal:
            frame_id = "map"
            ns = "goal"

            marker = self.create_marker(666,ns, Marker.SPHERE, [self.agent_list[0].goal.x,self.agent_list[0].goal.y,0.2], 
                [0.3,0.3,0.3], [1.0,0.0,0.0,1.0], frame_id, None)
            self.visualize_array.markers.append(marker)

    def update_trajectory(self):
        for i in range(self.num_agents):
            # Create a new pose stamped with the current position
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = "map"
            pose_stamped.pose.position.x = self.agent_list[i].position.x
            pose_stamped.pose.position.y = self.agent_list[i].position.y

            # Update the trajectory for the specified robot
            
            self.trajectory[i].poses.append(pose_stamped)

            traj_to_keep = 500
            if len(self.trajectory[i].poses) > traj_to_keep:
                self.trajectory[i].poses = self.trajectory[i].poses[-traj_to_keep:]

            self.trajectory[i].header.frame_id = "map"

            self.traj_pub_list[i].publish(self.trajectory[i])

    def visualize_connections(self):
        frame_id = "map"
        ns = "connections"

        points = []
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if self.A_matrix[i][j] == 1:
                    points.append([self.agent_list[i].position.x ,self.agent_list[i].position.y ,self.agent_list[i].position.z])
                    points.append([self.agent_list[j].position.x ,self.agent_list[j].position.y,self.agent_list[i].position.z])

        marker = self.create_marker(222,ns, Marker.LINE_LIST, [0.0,0.0,0.0], 
                [0.04,0.04,0.04], [0.0,0.0,1.0,1.0], frame_id, points)
        self.visualize_array.markers.append(marker)
    
    def visualize_formation(self):
        frame_id = "map"
        ns = "formation"

        points = []
        for i in range(self.num_agents-1):
            points.append([self.formation[i][0] + self.agent_list[0].position.x ,self.formation[i][1] +self.agent_list[0].position.y ,self.agent_list[i].position.z])
            points.append([self.formation[i+1][0] + self.agent_list[0].position.x ,self.formation[i+1][1] +self.agent_list[0].position.y ,self.agent_list[i].position.z])
        points.append([self.formation[-1][0] + self.agent_list[0].position.x ,self.formation[-1][1] +self.agent_list[0].position.y ,self.agent_list[i].position.z])
        points.append([self.formation[0][0] + self.agent_list[0].position.x ,self.formation[0][1] +self.agent_list[0].position.y ,self.agent_list[i].position.z])
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

    def load_map(self, yaml_path, image_path):
        with open(yaml_path, 'r') as file:
            map_data = yaml.safe_load(file)
        
        resolution = map_data['resolution']
        origin = map_data['origin']
        negate = map_data['negate']
        
        # Read and optionally negate the map image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if not negate:
            image = 255 - image

        # Flip the image vertically
        image = np.flipud(image)

        # Convert the image to an occupancy grid (-1: unknown, 0: free, 100: occupied)
        occupancy_map = (image / 255.0).flatten()
        occupancy_map = np.where(occupancy_map > map_data['occupied_thresh'], 100,
                                np.where(occupancy_map < map_data['free_thresh'], 0, -1))

        # Create and populate the OccupancyGrid message
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = "map"  # Make sure this matches your TF frames
        grid.info = MapMetaData()
        grid.info.resolution = resolution
        grid.info.width = image.shape[1]
        grid.info.height = image.shape[0]

        # Set origin
        pose = Pose()
        pose.position.x = float(origin[0])
        pose.position.y = float(origin[1])
        pose.position.z = 0.0
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        grid.info.origin = pose

        grid.data = occupancy_map.astype(int).tolist()

        return grid

def main(args=None):
    rclpy.init(args=args)

    cfn = CrazyFliesNode(3)

    rclpy.spin(cfn)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    cfn.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()