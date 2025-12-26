#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# If you want robot info later:
# from dobot_msgs_v3.msg import ToolVectorActual
# from dobot_msgs_v3.srv import JointMovJ

import cv2
import numpy as np


def calculateTransform(
    xh: float,
    yh: float,
    obj_h: float,
    table_size: float = 300.0,
    table_center_base=(600.0, 0.0, 0.0),
    clearance: float = 80.0,
):
    # table_center_base x, y, z  is distance of origin robot base and table
    
    """
    Convert homography coordinates (from GREEN corner) to
    robot base positions, using YOUR axis mapping:

      - table is a 300x300 mm square
      - green = corner (0,0)
      - center = (table_size/2, table_size/2)
      - you already found correct mapping:
            x_table = -y_center_raw
            y_table = -x_center_raw
    """

    C = table_size / 2.0

    # homography corner → center
    x_center_raw = xh - C
    y_center_raw = yh - C

    # your final mapping (swap + flip)
    x_table = -y_center_raw
    y_table = -x_center_raw

    bx, by, bz = table_center_base

    # table center → robot base
    X = bx + x_table
    Y = by + y_table

    Z_table = bz
    #Z_pick = Z_table + obj_h
    Z_pick = 15
    Z_approach = Z_table + clearance

    return (X, Y, Z_approach), (X, Y, Z_pick), {
        "x_center_raw": x_center_raw,
        "y_center_raw": y_center_raw,
        "x_table": x_table,
        "y_table": y_table,
        "Z_table": Z_table,
    }


class VisionPickNode(Node):
    def __init__(self):
        super().__init__('vision_pick_node')

        self.bridge = CvBridge()

        # ---- parameters ----
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('table_size', 300.0)
        self.declare_parameter('table_center_x', 600.0)
        self.declare_parameter('table_center_y', 0.0)
        self.declare_parameter('table_center_z', 0.0)
        self.declare_parameter('object_height', 10.0)
        self.declare_parameter('clearance', 80.0)
        
        self.declare_parameter('lower_red', [173, 69, 114])
        self.declare_parameter('upper_red', [179, 255, 198])

        self.declare_parameter('lower_blue', [99, 100, 85])
        self.declare_parameter('upper_blue', [124, 255, 180])

        self.declare_parameter('lower_black', [0, 0, 0])
        self.declare_parameter('upper_black', [180, 255, 70])
        
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        
        
        """
        =======================================================================================
            Self Parameter Set
        =======================================================================================
        """
        
        # Declare parameters with default values
        self.declare_parameter('showimg', 0)
        
        # Read parameter values
        self.showimg   = self.get_parameter('showimg').value
        
        
        
        # subscribe image
        self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        
        self.get_logger().info(f"VisionPickNode subscribed to {image_topic}")

    def centroid(self,cnt):
        """Return contour centroid (cx,cy) in pixels."""
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        
    def order_tl_tr_br_bl(self,pts):
        """
        Order 4 points into: TL, TR, BR, BL.
        Homography needs consistent ordering.
        """
        pts = np.array(pts, dtype=np.float32)
        s = pts.sum(axis=1)          # x+y
        d = np.diff(pts, axis=1)     # x-y

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32) 

    # ====== Your homography code should go here ======
    def homography_find_xy_mm(self, frame) -> (bool, float, float):
    
        
        """
        Dummy implementation for now.

        You ALREADY have code that:
          - finds the four corners
          - computes homography
          - returns X, Y in mm (green text on your screenshot)

        Replace content of this function with your real code and
        return (True, x_mm, y_mm) when you detect a point.
        """
        
        # ===================== CONFIG =====================
        MIN_RED_AREA = 100
        MIN_OBJ_AREA = 50
        ROI_ENABLE = True
        ROI_SHRINK_PIX = 15
        table_size = 300.00
        
        kernel = np.ones((5, 5), np.uint8)
        roi_shrink_kernel = np.ones((ROI_SHRINK_PIX, ROI_SHRINK_PIX), np.uint8) if ROI_SHRINK_PIX > 0 else None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red   = np.array(self.get_parameter('lower_red').value)
        upper_red   = np.array(self.get_parameter('upper_red').value)
        lower_blue  = np.array(self.get_parameter('lower_blue').value)
        upper_blue  = np.array(self.get_parameter('upper_blue').value)
        lower_black = np.array(self.get_parameter('lower_black').value)
        upper_black = np.array(self.get_parameter('upper_black').value)

        # ===================== DRAW CANVAS =====================
        debug = frame.copy()

        # ==========================================================
        # 🔴 1) Detect RED markers (each frame)
        # ==========================================================
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        red_pts = []
        for c in contours_red:
            if cv2.contourArea(c) > MIN_RED_AREA:
                cxy = self.centroid(c)
                if cxy is None:
                    continue
                cx, cy = map(int, cxy)
                red_pts.append([cx, cy])
                # draw red markers live
                cv2.drawContours(debug, [c], -1, (0,0,255), 2)
                cv2.circle(debug, (cx, cy), 6, (0,255,255), -1)


        # ==========================================================
        # 🟩 2) ROI from red points (if 4 detected)
        # ==========================================================
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if ROI_ENABLE and len(red_pts) == 4:
            red_pts = self.order_tl_tr_br_bl(np.array(red_pts)).astype(np.int32)
            cv2.fillConvexPoly(roi_mask, red_pts, 255)
            if roi_shrink_kernel is not None:
                roi_mask = cv2.erode(roi_mask, roi_shrink_kernel, iterations=1)

            # Draw ROI Border + transparent fill
            cv2.polylines(debug, [red_pts], True, (0,255,255), 1)
            #overlay = debug.copy()
            #cv2.fillConvexPoly(overlay, red_pts, (0,255,0))
            #cv2.addWeighted(overlay, 0.15, debug, 0.85, 0, debug)
        else:
            cv2.putText(debug, "NO ROI (need 4 red)", (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)


        # ==========================================================
        # 🔵 3) Object detect (Blue/Black) inside ROI
        # ==========================================================
        mask_blue  = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_obj = cv2.bitwise_or(mask_blue, mask_black)

        mask_obj = cv2.bitwise_and(mask_obj, roi_mask)
        mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_OPEN, kernel)
        mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_CLOSE, kernel)

        contours_obj, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        obj_center = None
        if len(contours_obj) > 0:
            # draw ALL detected objects (only once per frame, not twice)
            for c in contours_obj:
                if cv2.contourArea(c) > MIN_OBJ_AREA:
                    obj_center = self.centroid(c)
                    cx, cy = map(int, obj_center)
                    cv2.drawContours(debug, [c], -1, (255,0,0), 2)
                    cv2.circle(debug, (cx, cy), 4, (255,0,0), -1)

        else:
            cv2.putText(debug, "NO OBJECT", (40,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2)
            
        if len(red_pts) == 4 and obj_center is not None:
            red_pts = self.order_tl_tr_br_bl(red_pts)
            
            world_pts = np.array([
                [0, 0],
                [table_size, 0],
                [table_size, table_size],
                [0, table_size]
            ], dtype=np.float32)
            
            H, _ = cv2.findHomography(red_pts, world_pts)

            obj_px = np.array([[obj_center]], dtype=np.float32)
            obj_world = cv2.perspectiveTransform(obj_px, H)

            X, Y = obj_world[0][0]        
            cv2.putText(debug, f"OBJ ({X:.2f},{Y:.2f})", (cx+17, cy+17),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
            found = True
        else:
            X = 0.0
            Y = 0.0
            found = False 
        # ==========================================================
        # 🪟 4) FINAL COMBINED WINDOW
        # ==========================================================


        # Show live windows (resizable)
        if self.showimg == 0:
            cv2.imshow("Live Camera", frame)
            cv2.imshow("WIN_DEBUG", debug)
            
        elif self.showimg == 1:
            cv2.imshow("Live Camera", frame)

        
        elif self.showimg == 2:
            cv2.imshow("WIN_MASK", mask_obj_live)
            cv2.imshow("WIN_ROI", roi_mask)
            
        elif self.showimg == 3:
            cv2.imshow("Live Camera", frame)
            cv2.imshow("WIN_DEBUG", debug)
            cv2.imshow("WIN_MASK", mask_obj_live)
            cv2.imshow("WIN_ROI", roi_mask)       
        cv2.waitKey(1)

        
        
        
        return found, X, Y

    # ======================================
    def image_callback(self, msg: Image):
        # ROS Image -> OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 1) Run homography + object detection
        found, x_mm, y_mm = self.homography_find_xy_mm(frame)

        if not found:
            # Just show the image and return
            # cv2.imshow("Not Found", frame)
            # cv2.waitKey(1)
            return

        # 2) Get parameters
        table_size = float(self.get_parameter('table_size').value)
        bx = float(self.get_parameter('table_center_x').value)
        by = float(self.get_parameter('table_center_y').value)
        bz = float(self.get_parameter('table_center_z').value)
        obj_h = float(self.get_parameter('object_height').value)
        clearance = float(self.get_parameter('clearance').value)

        # 3) Convert to robot base coordinates
        approach, pick, debug = calculateTransform(
            xh=x_mm,
            yh=y_mm,
            obj_h=obj_h,
            table_size=table_size,
            table_center_base=(bx, by, bz),
            clearance=clearance
        )

        X_a, Y_a, Z_a = approach
        X_p, Y_p, Z_p = pick
    
        Cmd_str = "ros2 service call /dobot_bringup_v3/srv/MovL dobot_msgs_v3/srv/MovL"
        Approach_str = f'{Cmd_str} "{{x: {X_a:.2f}, y: {Y_a:.2f}, z: {Z_a:.2f}, rx: -180.0, ry: 0.0, rz: -90.0, param_value: []}}"'
        Pick_str = f'{Cmd_str} "{{x: {X_p:.2f}, y: {Y_p:.2f}, z: {Z_p:.2f}, rx: -180.0, ry: 0.0, rz: -90.0, param_value: []}}"'
     
        # 4) Print info so you can check against reality
        self.get_logger().info(
            f"🧡 Homo XY = ({x_mm:.2f}, {y_mm:.2f}) mm\n"
            f"{Approach_str}\n"
            f"{Pick_str}\n",throttle_duration_sec=5.0
        )

        # Optional: draw result on image
        cv2.circle(frame, (int(msg.width/2), int(msg.height/2)), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Xh={x_mm:.2f} Yh={y_mm:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        # cv2.imshow("camera", frame)
        # cv2.waitKey(1)

        # 5) LATER: here you will call Dobot service to move.
        #    For now, we ONLY compute and print positions.
        #    Example skeleton (DON'T enable until IK is ready):
        #
        # if self.joint_client is not None:
        #     req = JointMovJ.Request()
        #     req.j1 = ...
        #     ...
        #     self.joint_client.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = VisionPickNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

