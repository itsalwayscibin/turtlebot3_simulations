#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
from nav_msgs.msg import Odometry
import math
import time

class Nav2TurtleBotNavigation(Node):
    def __init__(self):
        super().__init__('nav2_turtlebot_navigation')
        
        # Initialize Nav2 navigator
        self.navigator = BasicNavigator()
        
        # Subscribers for position tracking
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.position_received = False
        
        # Tolerance settings
        self.position_tolerance = 0.1  # 10cm position tolerance
        self.yaw_tolerance = math.radians(10)  # 10 degrees yaw tolerance (adjustable)
        
        self.get_logger().info("Nav2 TurtleBot Navigation System Initialized")
        self.get_logger().info(f"Position tolerance: {self.position_tolerance:.2f}m")
        self.get_logger().info(f"Yaw tolerance: {math.degrees(self.yaw_tolerance):.1f}°")
        
    def odom_callback(self, msg):
        """Get robot position from odometry"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
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
    
    def check_goal_reached(self, target_x, target_y, target_yaw):
        """Check if robot has reached the goal within tolerances"""
        # Position check
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        position_error = math.sqrt(dx*dx + dy*dy)
        
        # Yaw check
        yaw_error = abs(self.normalize_angle(target_yaw - self.current_yaw))
        
        position_ok = position_error <= self.position_tolerance
        yaw_ok = yaw_error <= self.yaw_tolerance
        
        self.get_logger().info(f"Position error: {position_error:.3f}m (tolerance: {self.position_tolerance:.2f}m)")
        self.get_logger().info(f"Yaw error: {math.degrees(yaw_error):.1f}° (tolerance: {math.degrees(self.yaw_tolerance):.1f}°)")
        
        return position_ok and yaw_ok
    
    def set_tolerances(self, position_tol=0.1, yaw_tol_degrees=10.0):
        """Set position and yaw tolerances"""
        self.position_tolerance = position_tol
        self.yaw_tolerance = math.radians(yaw_tol_degrees)
        self.get_logger().info(f"Tolerances updated - Position: {position_tol:.2f}m, Yaw: {yaw_tol_degrees:.1f}°")
    
    def create_pose_stamped(self, x, y, yaw):
        """Create a PoseStamped message with absolute coordinates"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'  # Use 'map' for absolute coordinates
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        return pose
    
    def navigate_to_absolute_pose(self, target_x, target_y, target_yaw=0.0):
        """Navigate to an absolute pose using Nav2"""
        
        # Wait for Nav2 to be ready
        self.get_logger().info("⏳ Waiting for Nav2 to be ready...")
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("✅ Nav2 is ready!")
        
        # Wait for position data
        timeout = 10
        start_time = time.time()
        while not self.position_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if not self.position_received:
            self.get_logger().error("❌ No odometry data received!")
            return False
        
        # Show current position
        self.get_logger().info(f"📍 Current position: ({self.current_x:.2f}, {self.current_y:.2f}, {math.degrees(self.current_yaw):.1f}°)")
        
        # Check if already at target
        if self.check_goal_reached(target_x, target_y, target_yaw):
            self.get_logger().info("🎯 Already at target position within tolerance!")
            return True
        
        # Calculate distance to target
        distance = math.sqrt((target_x - self.current_x)**2 + (target_y - self.current_y)**2)
        yaw_diff = abs(self.normalize_angle(target_yaw - self.current_yaw))
        self.get_logger().info(f"📏 Distance to target: {distance:.2f}m")
        self.get_logger().info(f"🔄 Yaw difference: {math.degrees(yaw_diff):.1f}°")
        
        # Create absolute goal pose
        goal_pose = self.create_pose_stamped(target_x, target_y, target_yaw)
        
        self.get_logger().info(f"🎯 Navigating to absolute position: ({target_x:.2f}, {target_y:.2f}, {math.degrees(target_yaw):.1f}°)")
        
        # Send goal to Nav2
        self.navigator.goToPose(goal_pose)
        
        # Monitor navigation progress
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                distance_remaining = feedback.distance_remaining
                estimated_time = feedback.estimated_time_remaining
                self.get_logger().info(f"📐 Distance remaining: {distance_remaining:.2f}m, ETA: {estimated_time.sec}s")
            
            # Check current position and tolerance
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check if goal reached within tolerance (early termination)
            if self.check_goal_reached(target_x, target_y, target_yaw):
                self.get_logger().info("🎯 Goal reached within tolerance, stopping navigation!")
                self.navigator.cancelTask()
                break
            
            # Sleep to avoid overwhelming the system
            time.sleep(1.0)
        
        # Final check
        final_check = self.check_goal_reached(target_x, target_y, target_yaw)
        
        # Check result
        result = self.navigator.getResult()
        if result == BasicNavigator.TaskResult.SUCCEEDED or final_check:
            self.get_logger().info("🎉 Navigation completed successfully!")
            # Show final position
            self.get_logger().info(f"📍 Final position: ({self.current_x:.2f}, {self.current_y:.2f}, {math.degrees(self.current_yaw):.1f}°)")
            return True
        elif result == BasicNavigator.TaskResult.CANCELED and final_check:
            self.get_logger().info("🎯 Navigation canceled but goal reached within tolerance!")
            return True
        elif result == BasicNavigator.TaskResult.CANCELED:
            self.get_logger().warn("⚠️ Navigation was canceled!")
            return False
        elif result == BasicNavigator.TaskResult.FAILED:
            self.get_logger().error("❌ Navigation failed!")
            return False
        else:
            self.get_logger().error(f"❓ Unknown navigation result: {result}")
            return final_check
    
    def navigate_through_absolute_waypoints(self, waypoints):
        """Navigate through multiple absolute waypoints"""
        self.get_logger().info("⏳ Waiting for Nav2 to be ready...")
        self.navigator.waitUntilNav2Active()
        
        # Wait for position data
        timeout = 10
        start_time = time.time()
        while not self.position_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if not self.position_received:
            self.get_logger().error("❌ No odometry data received!")
            return False
        
        # Show current position
        self.get_logger().info(f"📍 Starting from: ({self.current_x:.2f}, {self.current_y:.2f})")
        
        # Convert waypoints to absolute positions
        goal_poses = []
        for i, (x, y, yaw) in enumerate(waypoints):
            goal_pose = self.create_pose_stamped(x, y, yaw)
            goal_poses.append(goal_pose)
            self.get_logger().info(f"Waypoint {i+1}: ({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f}°)")
        
        # Send waypoints to Nav2
        self.navigator.followWaypoints(goal_poses)
        
        # Monitor progress
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                current_waypoint = feedback.current_waypoint
                self.get_logger().info(f"📍 Currently navigating to waypoint: {current_waypoint + 1}")
            
            # Update current position
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(1.0)
        
        # Check result
        result = self.navigator.getResult()
        if result == BasicNavigator.TaskResult.SUCCEEDED:
            self.get_logger().info("🎉 All waypoints completed successfully!")
            return True
        else:
            self.get_logger().error(f"❌ Waypoint navigation failed: {result}")
            return False

def main(args=None):
    rclpy.init(args=args)
    
    # Create navigation node
    navigator_node = Nav2TurtleBotNavigation()
    
    try:
        # Set custom tolerances (optional)
        navigator_node.set_tolerances(
            position_tol=0.05,    # 5cm position tolerance
            yaw_tol_degrees=15.0  # 15 degrees yaw tolerance
        )
        
        # ABSOLUTE target coordinates (in map frame)
        target_x = 1.63  # Absolute X coordinate in map
        target_y = 0.75  # Absolute Y coordinate in map
        target_yaw = 0.0  # Absolute orientation (0 = facing forward)
        
        navigator_node.get_logger().info(f"🎯 Target set to ABSOLUTE position: ({target_x}, {target_y}, {math.degrees(target_yaw):.1f}°)")
        
        # Navigate directly to absolute target
        success = navigator_node.navigate_to_absolute_pose(target_x, target_y, target_yaw)
        
        if success:
            navigator_node.get_logger().info(f"🎉 Successfully reached ABSOLUTE target position ({target_x}, {target_y})!")
        else:
            navigator_node.get_logger().error("💥 Failed to reach target position!")
            
    except KeyboardInterrupt:
        navigator_node.get_logger().info("🛑 Navigation interrupted by user")
    
    finally:
        navigator_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()