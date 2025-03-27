import mecademicpy.robot as mdr




robot = mdr.Robot()
robot.Connect(address='192.168.0.100')


def activate_robot():
    robot.ActivateRobot()
    robot.Home()
    robot.MoveJoints(0,0,0,0,0,0)
    robot.SetCartLinVel(5)
    #robot.MovePose(190, 0, 308, 0, 90, 0)
    robot.SetTrf(0,0  ,0  , 0 , 90 , 90 )
    #robot.MovePose(190, 0, 308, 0, 90, 0)

def close_robot():
    robot.WaitIdle()
    robot.DeactivateRobot()
    robot.Disconnect()


def move_to_pose(x,y,z):
    robot.MovePose(x, y, z, 0, 0, 0)
  
def move_to_lin_trf(x,y,z):
    robot.MoveLinRelTRF(x, y, z, 0, 0, 0)


#activate_robot()
#move_to_pose()
#close_robot()    