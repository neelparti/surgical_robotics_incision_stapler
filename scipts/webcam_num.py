import cv2

def find_available_cameras(max_tested=10):
    available_cameras = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            available_cameras.append(i)
            cap.release()
    if not available_cameras:
        print("No cameras detected.")
    return available_cameras

available_cameras = find_available_cameras()