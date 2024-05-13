import rospy
import asyncio as aio
from pymavlink import mavutil
from std_srvs.srv import Empty
from sensor_msgs.msg import CompressedImage, Image


def callback(data) : 
    print(len(data.data))
    print(data.height, data.width, data.encoding, data.is_bigendian, data.step)

def mav_connect(udp_port) : 
    rospy.loginfo("connect mav")
    connection = mavutil.mavlink_connection('udp:localhost:'+str(udp_port))
    # command = 400
    # param1 = 0  # 1 to ARM, 0 to DISARM
    # param2 = 21196  # Custom parameter (set to whatever value you need)
    
    return connection

def run_mavsafety_kill(connection, id):
    rospy.loginfo("Force disarm")
    # command = 400
    # param1 = 0  # 1 to ARM, 0 to DISARM
    # param2 = 21196  # Custom parameter (set to whatever value you need)
    msg = connection.mav.command_long_encode(
        target_system=id,        # Target system ID
        target_component=1,     # Target component ID
        command=246,
        confirmation=0,
        param1=1,
        param2=0,
        param3=0,
        param4=0,
        param5=0,
        param6=0,
        param7=0
    )
    connection.mav.send(msg)
    
rospy.init_node('test')
#rospy.Subscriber("/iris_0/stereo_camera/left/image_raw/compressed", CompressedImage, callback)
#rospy.Subscriber("/iris_0/camera/depth/image_raw", Image, callback)
rate = rospy.Rate(1)
con = mav_connect(14540)
while not rospy.is_shutdown() : 
    #print(0)
    #rospy.wait_for_service('/gazebo/pause_physics')
    
    #rospy.ServiceProxy('/gazebo/pause_physics',Empty)
    #a = rospy.wait_for_message("/iris_0/camera/depth/image_raw",Image,timeout=5.0)
    #print(len(a.data))
    rospy.logwarn("run_mavsafety_kill 1")
    run_mavsafety_kill(con,1)
    rate.sleep()
    rospy.logwarn("run_mavsafety_kill 2")
    run_mavsafety_kill(con,2)
    rate.sleep()
    rospy.logwarn("run_mavsafety_kill 3")
    run_mavsafety_kill(con,3)
    rate.sleep()
    rospy.logwarn("run_mavsafety_kill 4")
    run_mavsafety_kill(con,4)
    rate.sleep()
