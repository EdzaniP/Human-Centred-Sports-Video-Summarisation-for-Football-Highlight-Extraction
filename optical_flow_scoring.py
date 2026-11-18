import cv2
import numpy as np

#function calculates how much movement exists
# inside a video clip using optical flow,the movement direction
# and movement magnitude between frames
def calculate_optical_flow_score(video_path, start_time, end_time, frame_skip=5):
    #open file 
    cap = cv2.VideoCapture(video_path)

#check if file opened properly
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

 #getting the videos frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)

#converting time into frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

#going to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

#storing previous grayscale frame     
    previous_gray = None
    flow_scores = []#storing optical flow scores

 #current frame tracker
    current_frame = start_frame

#looping through frames
    while current_frame < end_frame:
        success, frame = cap.read() #read next frame

 #stop if frames couldnt be read
        if not success:
            break

#processing only every Nth frame to improve speed.
        if current_frame % frame_skip == 0:
             #resizing frame for faster processing
            frame = cv2.resize(frame, (320, 180))
            #convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


#computing optical flow 
            if previous_gray is not None:
                 # Farneback dense optical flow,calculates motion vectors
                # for all pixels between frames
                flow = cv2.calcOpticalFlowFarneback(
                     # Previous grayscale frame
                    previous_gray,
                    # Current grayscale frame
                    gray,
                    # Initial flow estimate
                    None,
                     # Image pyramid scaling factor.
        # Each pyramid level is half the size of the previous level
        # Helps detect large motion at
        # multiple resolutions.
                    0.5,
                    # Number of pyramid levels for better large motion detection
                    3,
                   # winsize:Size of pixel neighbourhood used for motion estimation
                    15,
                     # iterations: Number of optimisation iterations
                    3,
                    #Size of neighbourhood used for
        # polynomial approximation.
         # Determines how local image motion
        # is modelled.
                    5,
                    # Standard deviation of Gaussian
        # smoothing applied before fitting
        # the polynomial model to help reduce nose while retaining meaningful movement information
                    1.2,
                    #no flags used
                    0
                )
                
                
#converting flow vectors 
#convert x/y motion vectors into magnitude (movement strength)and  angle (movement direction)
                magnitude, angle = cv2.cartToPolar(
                    flow[..., 0],
                    flow[..., 1]
                )

                score = np.mean(magnitude)#average movement magnitude across frame,higher value means more motion/activity
                flow_scores.append(score) #save frame motion score

            previous_gray = gray #save current frame as previous frame
#move to next frame
        current_frame += 1

    cap.release()#release video object

#handling empty scores
    if len(flow_scores) == 0:
        return 0

 #return average motion intensity
 #  across the entire clip.
    return float(np.mean(flow_scores))