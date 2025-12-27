#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from dobot_msgs_v3.srv import *




class TestDobotService(Node):
    def __init__(self):
        super().__init__("test_node")
        self.get_logger().info("🎉 Test node is running!")
        
        # ---- Create service clients ----
        self.movlio_client = self.create_client(MovLIO, "/dobot_bringup_v3/srv/MovLIO")
        self.do_client     = self.create_client(DO,     "/dobot_bringup_v3/srv/DO")
        self.sync_client   = self.create_client(Sync,   "/dobot_bringup_v3/srv/Sync")

        # Wait a bit for services
        self._wait_for_service(self.movlio_client, "MovLIO")
        self._wait_for_service(self.do_client,     "DO")
        self._wait_for_service(self.sync_client,   "Sync")
        print("\n")
        
        self.call_MovLIO(520.0, -37.5, 580.0, -180.0, 0.0, -90.0, param_list=['1,1,2,1','1,1,1,0'])
        self.call_sync()
        self.call_MovLIO(520.0, -37.5, 40.0, -180.0, 0.0, -90.0, param_list=['1,1,2,0','1,1,1,1'])      


    def _wait_for_service(self, client, name):
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f"Service {name} not available after 5s")
        else:
            self.get_logger().info(f"Service {name} is ready")       
            
        
    # ------------------------------------------------------------
    # 1) MovLIO: linear move + extra options via param_value[]
    # ------------------------------------------------------------
    def call_MovLIO(self, x, y, z, rx=-180.0, ry=0.0, rz=-90.0, param_list=None):
        """Call MovLIO once and wait for result."""
        if param_list is None:
            param_list = []

        req = MovLIO.Request()
        req.x = float(x)
        req.y = float(y)
        req.z = float(z)
        req.rx = float(rx)
        req.ry = float(ry)
        req.rz = float(rz)
        # IMPORTANT: param_value is string[]
        req.param_value = [str(s) for s in param_list]

        self.get_logger().info(
            f"Calling MovLIO: ({req.x:.1f},{req.y:.1f},{req.z:.1f}) "
            f"rx={req.rx}, ry={req.ry}, rz={req.rz}, param_value={req.param_value}\n"
        )

        future = self.movlio_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f"MovLIO response res={future.result().res}")
        else:
            self.get_logger().error("MovLIO call failed")
            
    # ------------------------------------------------------------
    # 2) DO: digital output ON/OFF
    #    ⚠️ You MUST adjust field names to match your `ros2 interface show dobot_msgs_v3/srv/DO`
    # ------------------------------------------------------------
    def call_do(self, index: int, status: int):
        """
        Example DO service: set digital output.
        Many Dobot DO services look like:
            int32 index
            int32 status
            ---
            int32 res
        But PLEASE confirm with `ros2 interface show dobot_msgs_v3/srv/DO`.
        """
        req = DO.Request()

        # 🔴 ADJUST THESE FIELD NAMES IF NEEDED
        # Use exactly the names you see from ros2 interface show.
        # For example, it might be:
        #   req.index = index
        #   req.status = status
        # or   req.id = index ; req.value = status
        req.index = int(index)
        req.status = int(status)

        self.get_logger().info(f"Calling DO: index={req.index}, status={req.status}")

        future = self.do_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f"DO response res={future.result().res}")
        else:
            self.get_logger().error("DO call failed")

    # ------------------------------------------------------------
    # 3) Sync: block until robot finishes (depends on your driver)
    # ------------------------------------------------------------
    def call_sync(self):
        """
        Example Sync service call.
        You MUST check `ros2 interface show dobot_msgs_v3/srv/Sync` to see
        if it has any request fields. Often it's empty or has string[] param_value.
        """
        req = Sync.Request()

        # If Sync has fields, set them here, example:
        #   req.param_value = []
        #   or req.id = 0
        # Adjust to match your interface!

        self.get_logger().info("Calling Sync (wait robot)")

        future = self.sync_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f"Sync response res={future.result().res}")
        else:
            self.get_logger().error("Sync call failed")           
            
            
            
def main(args=None):
    rclpy.init(args=args)
    node = TestDobotService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()

