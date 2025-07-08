#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as ROSPoint
from rclpy.qos import qos_profile_sensor_data
from turtlebot_hd.srv import Dock
import math
import time
import numpy as np
import cv2
from enum import Enum

class WorkflowState(Enum):
    IDLE = 0
    NAVIGATING_TO_POSE = 1
    LIDAR_PROCESSING = 2
    COMPLETED = 3

class UnifiedTurtleBotController(Node):
    def __init__(self):
        super().__init__('unified_turtlebot_controller')
        
        # Workflow state - Start in IDLE
        self.current_state = WorkflowState.IDLE
        
        # Navigation components
        self.navigator = BasicNavigator()
        
        # Service for workflow trigger
        self.workflow_service = self.create_service(Dock, 'start_workflow', self.handle_workflow_request)
        
        # Service response tracking
        self.current_response = None
        self.service_in_progress = False
        
        # Subscriptions
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.scan_sub2 = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.distance_pub = self.create_publisher(Float32, 'distance_to_intersection', 10)
        self.angle_pub = self.create_publisher(Float32, 'angle_target', 10)
        
        # Navigation variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.position_received = False
        self.position_tolerance = 0.1
        self.yaw_tolerance = math.radians(10)
        self.navigation_started = False
        self.navigation_complete = False
        
        # Target pose
        self.target_x = -3.80 #1.54
        self.target_y = -1.90 #0.75
        self.target_yaw = -2.27
        
        # LiDAR processing variables
        self.lidar_target_yaw = None
        self.target_distance = None
        self.points = np.empty((0, 2))
        self.smooth_intersection = None
        self.smooth_alpha = 0.2
        self.scale = 200.0
        self.img_size = 400
        self.kp = 1.5
        
        # Workflow result tracking
        self.workflow_success = False
        self.workflow_message = ""
        
        # Timers
        self.create_timer(0.1, self.workflow_update)
        self.create_timer(0.1, self.update_plot)
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("🚀 Unified TurtleBot Controller initialized")
        self.get_logger().info("⏳ Waiting for service request...")
        self.get_logger().info("💡 Use: ros2 service call /start_workflow turtlebot_hd/srv/Dock")

    def handle_workflow_request(self, request, response):
        """Handle workflow service request"""
        if self.service_in_progress:
            self.get_logger().warn("⚠️ Workflow already in progress!")
            response.success = False
            response.message = "Workflow already running, please wait for completion"
            return response
        
        self.get_logger().info("📞 Workflow service request received!")
        self.get_logger().info("🚢 Starting navigation workflow...")
        
        # Reset workflow state
        self.reset_workflow_variables()
        self.service_in_progress = True
        self.current_response = response
        self.current_state = WorkflowState.NAVIGATING_TO_POSE
        
        # Don't return response yet - will be set when workflow completes
        return response

    def reset_workflow_variables(self):
        """Reset all workflow variables for new run"""
        self.navigation_started = False
        self.navigation_complete = False
        self.lidar_target_yaw = None
        self.target_distance = None
        self.points = np.empty((0, 2))
        self.smooth_intersection = None
        self.workflow_success = False
        self.workflow_message = ""

    def workflow_update(self):
        """Main workflow state machine"""
        if self.current_state == WorkflowState.NAVIGATING_TO_POSE:
            if not self.navigation_started and self.position_received:
                self.start_navigation()
            elif self.navigation_complete:
                self.get_logger().info("✅ Navigation completed, starting LiDAR processing...")
                self.current_state = WorkflowState.LIDAR_PROCESSING
        
        elif self.current_state == WorkflowState.COMPLETED:
            self.complete_workflow()

    def complete_workflow(self):
        """Complete the workflow and send service response"""
        if self.service_in_progress and self.current_response is not None:
            self.current_response.success = self.workflow_success
            self.current_response.message = self.workflow_message
            
            self.get_logger().info(f"🎉 Workflow completed: {self.workflow_message}")
            self.get_logger().info("⏳ Ready for next service request...")
            
            # Reset for next request
            self.service_in_progress = False
            self.current_response = None
            self.current_state = WorkflowState.IDLE

    # Navigation methods
    def odom_callback(self, msg):
        """Update current robot position"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        )
        self.position_received = True

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def check_goal_reached(self):
        """Check if robot reached the goal"""
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        position_error = math.sqrt(dx*dx + dy*dy)
        yaw_error = abs(self.normalize_angle(self.target_yaw - self.current_yaw))
        position_ok = position_error <= self.position_tolerance
        yaw_ok = yaw_error <= self.yaw_tolerance
        return position_ok and yaw_ok

    def create_pose_stamped(self, x, y, yaw):
        """Create PoseStamped message"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    def start_navigation(self):
        """Start navigation to predefined pose"""
        self.get_logger().info("⏳ Waiting for Nav2 to be ready...")
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("✅ Nav2 is ready!")
        
        if self.check_goal_reached():
            self.get_logger().info("🎯 Already at target position!")
            self.navigation_complete = True
            return
        
        goal_pose = self.create_pose_stamped(self.target_x, self.target_y, self.target_yaw)
        self.get_logger().info(f"🎯 Navigating to: ({self.target_x:.2f}, {self.target_y:.2f}, {math.degrees(self.target_yaw):.1f}°)")
        self.navigator.goToPose(goal_pose)
        self.navigation_started = True
        
        # Monitor navigation in separate thread-like approach
        self.create_timer(1.0, self.check_navigation_status)

    def check_navigation_status(self):
        """Check navigation status periodically"""
        if self.current_state != WorkflowState.NAVIGATING_TO_POSE or not self.navigation_started:
            return
            
        if self.navigator.isTaskComplete():
            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED or self.check_goal_reached():
                self.get_logger().info("🎉 Navigation completed successfully!")
                self.navigation_complete = True
            else:
                self.get_logger().error(f"❌ Navigation failed with result: {result}")
                self.workflow_success = False
                self.workflow_message = f"Navigation failed: {result}"
                self.current_state = WorkflowState.COMPLETED
        elif self.check_goal_reached():
            self.get_logger().info("🎯 Goal reached within tolerance!")
            self.navigator.cancelTask()
            self.navigation_complete = True

    # LiDAR processing methods
    def imu_callback(self, msg):
        """Update current yaw from IMU during LiDAR processing"""
        if self.current_state == WorkflowState.LIDAR_PROCESSING:
            q = msg.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def lidar_callback(self, msg):
        """Process LiDAR scan data"""
        if self.current_state != WorkflowState.LIDAR_PROCESSING:
            return
        
        angs = msg.angle_min + np.arange(len(msg.ranges))*msg.angle_increment
        rngs = np.array(msg.ranges)
        mask = np.isfinite(rngs) & (rngs > msg.range_min) & (rngs < msg.range_max)
        angs, rngs = angs[mask], rngs[mask]
        x = rngs * np.cos(angs)
        y = rngs * np.sin(angs)
        self.points = np.vstack((x,y)).T

    def scan_callback(self, msg):
        """Get target distance from scan"""
        if self.current_state == WorkflowState.LIDAR_PROCESSING:
            num_points = round(len(msg.ranges)/2)
            self.target_distance = np.mean(msg.ranges[0:5])

    def find_two_longest_lines(self, pts_sq):
        """Find two longest perpendicular lines"""
        if pts_sq.size == 0:
            return None
        
        canvas = np.zeros((self.img_size,self.img_size), dtype=np.uint8)
        for x,y in pts_sq:
            px = int(self.img_size/2 + x*self.scale)
            py = int(self.img_size/2 - y*self.scale)
            if 0<=px<self.img_size and 0<=py<self.img_size:
                canvas[py,px] = 255
        
        edges = cv2.Canny(canvas, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=30, maxLineGap=5)
        if lines is None or len(lines) < 2:
            return None
        
        segs = [l[0] for l in lines]
        segs.sort(key=lambda l: -((l[2]-l[0])**2 + (l[3]-l[1])**2))
        
        for i in range(len(segs)):
            for j in range(i+1,len(segs)):
                a1 = math.atan2(segs[i][3]-segs[i][1], segs[i][2]-segs[i][0])
                a2 = math.atan2(segs[j][3]-segs[j][1], segs[j][2]-segs[j][0])
                if abs(abs(a1-a2) - math.pi/2) < math.radians(15):
                    return [segs[i], segs[j]]
        return None

    def intersect(self, L1, L2):
        """Calculate intersection of two lines"""
        x1,y1,x2,y2 = L1; x3,y3,x4,y4 = L2
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-6:
            return None
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
        return (px, py)

    def ema_smooth(self, prev, new, alpha):
        """Exponential moving average smoothing"""
        if prev is None:
            return new
        return (alpha * new[0] + (1 - alpha) * prev[0],
                alpha * new[1] + (1 - alpha) * prev[1])

    def update_plot(self):
        """Main LiDAR processing and visualization"""
        if self.current_state != WorkflowState.LIDAR_PROCESSING or self.points.size == 0:
            return
        
        marker_array = MarkerArray()
        marker_id = 0
        
        # LiDAR points marker
        pts_marker = Marker()
        pts_marker.header.frame_id = 'base_link'
        pts_marker.type = Marker.SPHERE_LIST
        pts_marker.action = Marker.ADD
        pts_marker.id = marker_id; marker_id += 1
        pts_marker.scale.x = pts_marker.scale.y = pts_marker.scale.z = 0.01
        pts_marker.color.r = 0.0; pts_marker.color.g = 0.0; pts_marker.color.b = 1.0; pts_marker.color.a = 1.0
        pts_marker.points = [ROSPoint(x=float(x), y=float(y), z=0.0) for x, y in self.points]
        marker_array.markers.append(pts_marker)
        
        # Detection square marker
        square_marker = Marker()
        square_marker.header.frame_id = 'base_link'
        square_marker.type = Marker.LINE_STRIP
        square_marker.action = Marker.ADD
        square_marker.id = marker_id; marker_id += 1
        square_marker.scale.x = 0.01
        square_marker.color.r = 1.0; square_marker.color.g = 0.0; square_marker.color.b = 0.0; square_marker.color.a = 1.0
        sq_pts = [(0.0,-0.35,0.0), (0.0,0.35,0.0), (0.7,0.35,0.0), (0.7,-0.35,0.0), (0.0,-0.35,0.0)]
        square_marker.points = [ROSPoint(x=float(x), y=float(y), z=float(z)) for x, y, z in sq_pts]
        marker_array.markers.append(square_marker)
        
        # Filter points in detection square
        sq_mask = (
            (self.points[:,0] >= 0) & (self.points[:,0] <= 0.7) &
            (self.points[:,1] >= -0.35) & (self.points[:,1] <= 0.35)
        )
        pts_sq = self.points[sq_mask]
        
        # Find lines and intersection
        segs = self.find_two_longest_lines(pts_sq)
        if segs is not None:
            lines = []
            for x1p,y1p,x2p,y2p in segs:
                x1 = (x1p - self.img_size/2) / self.scale
                y1 = (self.img_size/2 - y1p) / self.scale
                x2 = (x2p - self.img_size/2) / self.scale
                y2 = (self.img_size/2 - y2p) / self.scale
                lines.append((x1,y1,x2,y2))
            
            # Line markers
            if lines:
                line_marker = Marker()
                line_marker.header.frame_id = 'base_link'
                line_marker.type = Marker.LINE_LIST
                line_marker.action = Marker.ADD
                line_marker.id = marker_id; marker_id += 1
                line_marker.scale.x = 0.02
                line_marker.color.r = 0.0; line_marker.color.g = 1.0; line_marker.color.b = 0.0; line_marker.color.a = 1.0
                line_marker.points = []
                for l in lines:
                    line_marker.points += [ROSPoint(x=l[0], y=l[1], z=0.0), ROSPoint(x=l[2], y=l[3], z=0.0)]
                marker_array.markers.append(line_marker)
            
            # Find intersection
            if len(lines) == 2:
                intersection = self.intersect(lines[0], lines[1])
                if intersection:
                    # Apply smoothing
                    if self.smooth_intersection is None:
                        self.smooth_intersection = (float(intersection[0]), float(intersection[1]))
                    else:
                        self.smooth_intersection = self.ema_smooth(
                            self.smooth_intersection, 
                            (float(intersection[0]), float(intersection[1])), 
                            self.smooth_alpha)
                    
                    # Intersection marker
                    inter_marker = Marker()
                    inter_marker.header.frame_id = 'base_link'
                    inter_marker.type = Marker.SPHERE
                    inter_marker.action = Marker.ADD
                    inter_marker.id = marker_id; marker_id += 1
                    inter_marker.scale.x = inter_marker.scale.y = inter_marker.scale.z = 0.04
                    inter_marker.color.r = 1.0; inter_marker.color.g = 0.0; inter_marker.color.b = 1.0; inter_marker.color.a = 1.0
                    inter_marker.pose.position.x = self.smooth_intersection[0]
                    inter_marker.pose.position.y = self.smooth_intersection[1]
                    marker_array.markers.append(inter_marker)
                    
                    # Connection line marker
                    conn_marker = Marker()
                    conn_marker.header.frame_id = 'base_link'
                    conn_marker.type = Marker.LINE_STRIP
                    conn_marker.action = Marker.ADD
                    conn_marker.id = marker_id; marker_id += 1
                    conn_marker.scale.x = 0.01
                    conn_marker.color.r = 1.0; conn_marker.color.g = 0.0; conn_marker.color.b = 1.0; conn_marker.color.a = 1.0
                    conn_marker.points = [
                        ROSPoint(x=0.0, y=0.0, z=0.0), 
                        ROSPoint(x=float(self.smooth_intersection[0]), y=float(self.smooth_intersection[1]), z=0.0)
                    ]
                    marker_array.markers.append(conn_marker)
                    
                    # Publish distance and angle
                    distance_to_intersection = math.hypot(self.smooth_intersection[0], self.smooth_intersection[1])
                    self.distance_pub.publish(Float32(data=distance_to_intersection))
                    
                    if self.current_yaw is not None:
                        angle = self.current_yaw + math.atan2(self.smooth_intersection[1], self.smooth_intersection[0])
                        self.lidar_target_yaw = angle
                        self.angle_pub.publish(Float32(data=angle))
        
        self.marker_pub.publish(marker_array)

    def control_loop(self):
        """Robot control loop"""
        if self.current_state != WorkflowState.LIDAR_PROCESSING:
            return
        
        if self.lidar_target_yaw is None or self.target_distance is None:
            return
        
        if self.lidar_target_yaw is None or self.current_yaw is None:
            return
        
        error = self.lidar_target_yaw - self.current_yaw
        
        if abs(error) < 0.0174533:  # ~1 degree
            self.get_logger().info("Yaw error is within threshold, stopping rotation.")
            if self.target_distance < 0.28:
                self.get_logger().info("Target distance reached, stopping movement.")
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                
                # Set successful completion
                self.workflow_success = True
                self.workflow_message = f"Workflow completed successfully! Reached target at distance {self.target_distance:.3f}m"
                self.current_state = WorkflowState.COMPLETED
                return
            else:
                self.get_logger().info(f"Remaining distance - {self.target_distance:.3f}.")
                twist = Twist()
                twist.linear.x = 0.2 * self.target_distance
                self.cmd_pub.publish(twist)
            return
        
        control_signal = self.kp * error
        twist = Twist()
        twist.angular.z = control_signal
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"Current Yaw: {self.current_yaw:.2f}, {self.lidar_target_yaw}, Err: {error}, Control Signal: {twist.angular.z:.2f}")

def main():
    rclpy.init()
    
    try:
        controller = UnifiedTurtleBotController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("🛑 Controller interrupted by user")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()