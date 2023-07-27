from ctypes import sizeof
from tkinter import W
import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import os




def IncreaseContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    #result = np.hstack((img, enhanced_img))
    return enhanced_img

def roiImageFromHand( path_out_img, option, cap):
    # For webcam input:
    #cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    # cap.set(3, 640)
    # cap.set(4, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # open = ControlDoor.Detected_Object()
    # frame_count = 0
    # start_time = time.time()

    with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
           )       as hands:
        while cap.isOpened():
            if (option == 1): # option 1 is data collection
                valueOfImage = len([entry for entry in os.listdir(path_out_img) if os.path.isfile(os.path.join(path_out_img, entry))]) + 1
                # print("self.valueOfImage",valueOfImage)
                if (valueOfImage <= 10):
                    success, image = cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        continue
                    try:
                        # frame_count += 1

                        imgaeResize = IncreaseContrast(image)

                        # imgaeRGB = imgaeResize
                        # imgaeResize.flags.writeable = True
                        # imgaeRGB.flags.writeable = True
                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands.process(imgaeResize)
                        # print(results)

                        cropped_image = cv2.cvtColor(imgaeResize, cv2.COLOR_BGR2GRAY)

                        h = cropped_image.shape[0]
                        w = cropped_image.shape[1]
                        if results.multi_hand_landmarks:

                            # loop for get poin 5 9 13 15
                            for hand_landmark in results.multi_hand_landmarks:
                                pixelCoordinatesLandmarkPoint5 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[5].x, hand_landmark.landmark[5].y, w, h)
                                pixelCoordinatesLandmarkPoint9 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[9].x, hand_landmark.landmark[9].y, w, h)
                                pixelCoordinatesLandmarkPoint13 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[13].x, hand_landmark.landmark[13].y, w, h)
                                pixelCoordinatesLandmarkPoint17 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[17].x, hand_landmark.landmark[17].y, w, h)

                                x1 = (pixelCoordinatesLandmarkPoint17[0] +  pixelCoordinatesLandmarkPoint13[0]) / 2 + (pixelCoordinatesLandmarkPoint17[0] - pixelCoordinatesLandmarkPoint13[0])
                                y1 = (pixelCoordinatesLandmarkPoint17[1] + pixelCoordinatesLandmarkPoint13[1]) / 2 - 50
                                x2 = (pixelCoordinatesLandmarkPoint5[0] + pixelCoordinatesLandmarkPoint9[0]) / 2 - 50
                                y2 = (pixelCoordinatesLandmarkPoint5[1] + pixelCoordinatesLandmarkPoint9[1]) / 2 - 50
                                #sau khi cos 4 diem
                                #h, w = cropped_image.shape
                                theta = np.arctan2((y2 - y1), (x2 - x1))*180/np.pi 


                                R = cv2.getRotationMatrix2D(
                                    (int(x2), int(y2)), theta, 1)

                                #print(int(x2), int(y2))
                                #print("R", R)
                                align_img = cv2.warpAffine(cropped_image, R, (w, h)) 
                                imgaeRGB = cv2.warpAffine(imgaeRGB, R, (w, h)) 

                                #cv2.imshow("imgaeRGB", imgaeRGB)

                        results = hands.process(imgaeRGB)
                        # print(results)

                        cropped_image = cv2.cvtColor(imgaeRGB, cv2.COLOR_BGR2GRAY)

                        h = cropped_image.shape[0]
                        w = cropped_image.shape[1]

                        print("toi day co duoc khong vay ?")
                        if results.multi_hand_landmarks:

                            # loop for get poin 5 9 13 15
                            for hand_landmark in results.multi_hand_landmarks:
                                pixelCoordinatesLandmarkPoint5 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[5].x, hand_landmark.landmark[5].y, w, h)
                                pixelCoordinatesLandmarkPoint9 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[9].x, hand_landmark.landmark[9].y, w, h)
                                pixelCoordinatesLandmarkPoint13 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[13].x, hand_landmark.landmark[13].y, w, h)
                                pixelCoordinatesLandmarkPoint17 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[17].x, hand_landmark.landmark[17].y, w, h)
                            


                                x1 = (pixelCoordinatesLandmarkPoint17[0] +  pixelCoordinatesLandmarkPoint13[0]) / 2 
                                y1 = (pixelCoordinatesLandmarkPoint17[1] + pixelCoordinatesLandmarkPoint13[1]) / 2 
                                x2 = (pixelCoordinatesLandmarkPoint5[0] + pixelCoordinatesLandmarkPoint9[0]) / 2 
                                y2 = (pixelCoordinatesLandmarkPoint5[1] + pixelCoordinatesLandmarkPoint9[1]) / 2 
                                #sau khi cos 4 diem
                                #h, w = cropped_image.shape
                                theta = np.arctan2((y2 - y1), (x2 - x1))*180/np.pi 
                                print(theta)
                                R = cv2.getRotationMatrix2D(
                                    (int(x2), int(y2)), theta, 1)

                                align_img = cv2.warpAffine(cropped_image, R, (w, h)) 

                                

                                point_1 = [x1, y1]
                                point_2 = [x2, y2]


                                point_1 = (R[:, :2] @ point_1 + R[:, -1]).astype(np.int32)
                                point_2 = (R[:, :2] @ point_2 + R[:, -1]).astype(np.int32)
                                
                                # bien doi
                                landmarks_selected_align = {
                                    "x": [point_1[0], point_2[0]], "y": [point_1[1], point_2[1]]}

                                point_1 = np.array([landmarks_selected_align["x"]
                                            [0], landmarks_selected_align["y"][0]])
                                point_2 = np.array([landmarks_selected_align["x"]
                                                    [1], landmarks_selected_align["y"][1]])
                          

                                uxROI = pixelCoordinatesLandmarkPoint17[0]
                                uyROI = pixelCoordinatesLandmarkPoint17[1]
                                lxROI = pixelCoordinatesLandmarkPoint5[0]
                                lyROI = point_2[1] + 4*(point_2-point_1)[0]//3 
                                # print("lyROI", lyROI)

                                # cv2.rectangle(roi_zone_img, (lxROI, lyROI),
                                #     (uxROI, uyROI), (0, 255, 0), 2)

                                # cv2.imshow("roi_zone_img", roi_zone_img)
                                # 
                                # valueOfImage = len([entry for entry in os.listdir(path) if os.path.isfile(os.path.join(path, entry))]) + 1
                                


                                roi_img = align_img[uyROI:lyROI, uxROI:lxROI]


                                roi_img = cv2.resize(roi_img, (128, 128))

                                    # self.valueOfImage = len([entry for entry in os.listdir(path_out_img) if os.path.isfile(os.path.join(path_out_img, entry))]) + 1
                                path = path_out_img + "/0001_000" + str(valueOfImage) + ".bmp"

                                    # cv2.rectangle(roi_zone_img, (lx, ly),
                                    #             (ux, uy), (0, 255, 0), 2)

                            
                                # roi_img = cv2.resize(roi_img, (128, 128))
                                cv2.imwrite(path, roi_img)
                                
                                
                        if cv2.waitKey(5) & 0xFF == 27:
                            break

                    except:
                        print("loi ROI")
                else:
                    cap.release()
                    break
            
        cv2.destroyAllWindows()
        return 1

roiImageFromHand(path_out_img='./ROI1/',option=1,cap=cv2.VideoCapture(0))
