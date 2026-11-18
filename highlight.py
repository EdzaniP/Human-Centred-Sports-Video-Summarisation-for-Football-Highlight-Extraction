import cv2

#function for selecting highest scoring clips but avoiding clips that are too close together
#to reduce redundancy
def select_top_clips(clip_scores, max_clips=10, min_gap_seconds=10):
    
    #Selects the best clips while avoiding clips that are too close together.
    

    # Sort clips by score from highest to lowest
    sorted_clips = sorted(
        clip_scores,
        key=lambda clip: clip["score"],
        reverse=True #in descending order from largest score to smallest score
    )

#Storing final selected highlight clips
    selected_clips = []

    for clip in sorted_clips:
        #just assumingthat clip is not too close initially
        too_close = False
#checking distance of selected clips 
        for selected in selected_clips:
            #calculating time gap between clips
            time_gap = abs(clip["start"] - selected["start"])
            
            # If clips are too close together,
            # skip this clip
            if time_gap < min_gap_seconds:
                too_close = True
                break
#if not too close add clip
        if not too_close:
            selected_clips.append(clip)
            
#stop if maximum number of clips have been reached
        if len(selected_clips) == max_clips:
            break

    # Sort selected clips back into chronological order
    selected_clips = sorted(
        selected_clips,
        key=lambda clip: clip["start"]
    )

    return selected_clips

#function combines all selected highlight clips into one final summary video.
def create_highlight_video(video_path, selected_clips, output_path):
    """
    Creates one final highlight video from the selected clips.
    """
#open input video
    cap = cv2.VideoCapture(video_path)
#checkig if file was opened succesfully
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

#getting video properties
    fps = cap.get(cv2.CAP_PROP_FPS)#getting the frames per second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))#getting the video width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))#getting the video height

#defining what type of output the video will be in
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#creating the video output
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

#looping through the selected highlight clips
    for clip in selected_clips:
        
        #converting timestamps into frame numbers
        start_frame = int(clip["start"] * fps)
        end_frame = int(clip["end"] * fps)

        #going to to videos start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame #tracking current frame position

        while current_frame < end_frame:
           #read next frame
            success, frame = cap.read()
#if frame cant be read
            if not success:
                break
# writing frame into final highlight video
            writer.write(frame)
            current_frame += 1# Move to next frame

#realeasing video objects to free memory
    cap.release()
    writer.release()