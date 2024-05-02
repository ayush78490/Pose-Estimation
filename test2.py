import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

count=0
# count_incremented = False 


def calculate_angle(a, b, c):
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    #text on the screen
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50, 50) 
    fontScale = 1
    color = (255, 0, 0)  
    thickness = 2
    image = image = cv2.putText(image, 'condition : ' + str(count), org, font, fontScale, color, thickness, cv2.LINE_AA)




    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        #RealTimeAngle
        def biseps_angle(a, b, c):
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            return angle


        # left_wrist= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
        # right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        # print(left_wrist, right_wrist)

        #left Hand
        left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        left_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        left_wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y]
        
        angle_left_arm = calculate_angle(left_shoulder, left_elbow, left_wrist)

        cv2.putText(image, str(int(angle_left_arm)), 
                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                   )
        bisepsLeftAngle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        #Right biseps------------------------->
        
        #Right Hand
        right_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        right_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        right_wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        
        angle_right_arm = calculate_angle(right_shoulder, right_elbow, right_wrist)

        cv2.putText(image, str(int(angle_right_arm)), 
                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                   )
        
        
        
        #Right Hip
        right_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        right_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y]
        right_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        
        angle_right_hip = calculate_angle(right_shoulder, right_hip, right_knee)

        cv2.putText(image, str(int(angle_right_hip)), 
                    tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                   )
       #Left Hip
        left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        left_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y]
        left_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y]
        
        angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)

        cv2.putText(image, str(int(angle_left_hip)), 
                    tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                   )
        
        #Left shoulder
        left_elbow= [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        left_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y]
        
        angle_left_shoulder = calculate_angle(left_elbow, left_shoulder, left_hip)
        # print(angle_left_shoulder)
        cv2.putText(image, str(int(angle_left_shoulder)), 
                    tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                   )
        
        #Right shoulder
        right_elbow= [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        right_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        right_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y]
        
        angle_right_shoulder = calculate_angle(right_elbow, right_shoulder, right_hip)

        cv2.putText(image, str(int(angle_right_shoulder)), 
                    tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                   )
        
        #----------------> Push Up <----------------------------
        if 60 <= bisepsLeftAngle <= 90:
            cv2.putText(image, str(int(angle_left_arm)),
            tuple(np.multiply(left_elbow,[640, 440]).astype(int)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2, cv2.LINE_AA
            )
        



        #----------------> Straight Bisepes <----------------------------
        if angle_left_shoulder>90:
            cv2.putText(image, str(int(angle_left_arm)),
            tuple(np.multiply(left_elbow,[640, 440]).astype(int)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
            )
        if angle_right_shoulder>90:
            cv2.putText(image, str(int(angle_right_arm)),
            tuple(np.multiply(right_elbow,[640, 440]).astype(int)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
            )
            count=("working good")
        else:
            count=("not working ")



        #----------------> Left Bisepes <----------------------------
        if 30 <= bisepsLeftAngle <= 60:
            cv2.putText(image, str(int(angle_left_arm)),
            tuple(np.multiply(left_elbow,[640, 440]).astype(int)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
            )
        #     count=("working good")
        # else:
        #     count=("not working ")

        #------------------>Right Bisepes <-----------------------------

        bisepsRightAngle=int(biseps_angle(right_shoulder, right_elbow, right_wrist))
        if 30<=bisepsRightAngle<=60:
            cv2.putText(image, str(int(angle_right_arm)),
            tuple(np.multiply(right_elbow,[640, 440]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA 
            )
            count=("working good")
        else:
            count=("not working ")
       
        #--------------->feet touching exersise <-------------------------
        if angle_right_hip<90 :                           
            cv2.putText(image, str(int(angle_right_hip)),
            tuple(np.multiply(right_hip,[640, 440]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA 
            )
        #     count=("working good")
        # else:
        #     count=("not working ")
        
        
    

    cv2.imshow('AI Gym Trainer', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
