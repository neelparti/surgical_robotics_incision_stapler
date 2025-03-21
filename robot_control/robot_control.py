import mecademicpy.robot as mdr
robot = mdr.Robot()

#this needs to be changed for our IP
robot.Connect(address='192.168.0.100')


def activate_robot():
    robot.ActivateRobot()
    robot.Home()

def close_robot():
    robot.WaitIdle()
    robot.DeactivateRobot()
    robot.Disconnect()