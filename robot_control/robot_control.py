import mecademicpy.robot as mdr




robot = mdr.Robot()
robot.Connect(address='192.168.0.100')


def activate_robot():
    robot.ActivateRobot()
    robot.Home()
    robot.MoveJoints(0,0,0,0,0,0)#home robot
    robot.SetCartLinVel(10)#slow it down
    #robot.SetBlending
    #robot.MovePose(190, 0, 308, 0, 90, 0)
    robot.SetTrf(0,0 ,0 , 0 , 90 , 90 )# change trf
    #robot.MovePose(190, 0, 308, 0, 90, 0)

def close_robot():
    robot.WaitIdle()
    robot.DeactivateRobot()
    robot.Disconnect()


def move_to_pose(x,y,z):
    robot.MovePose(x, y, z, 0, 0, 0)
  
def move_to_lin_trf(x,y,z):
    robot.MoveLinRelTRF(x, y, z, 0, 0, 0)


def move_to_defined():
        robot.MovePose(190, 0, 124.725, -180, 0, -90)
    


#activate_robot()
#move_to_pose()
#close_robot()    