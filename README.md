# Starting
### 1. Clone git to dobot_ws/src and build it
``` 
cd ~/dobot_ws/src
git clone https://github.com/RubyGems19/dobot_vision_pick_v1.git
```
### 2. Build
```
cd ~/dobot_ws
colcon build --packages-select dobot_vision_pick
```
### 2. Run
showimg paramter
 - 0 show live camera and debug camera
 - 1 show live camera and debug camera
 - 2 show mask obj and ROI
 - 3 show all frme
```
source ~/.bashrc
ros2 run dobot_vision_pick vision_pick_node --ros-args -p showimg:=0
```
