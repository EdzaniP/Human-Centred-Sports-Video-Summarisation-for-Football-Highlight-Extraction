import cv2
import os
import time

from scoring import calculate_motion_score
from optical_flow_scoring import calculate_optical_flow_score
from highlight import select_top_clips, create_highlight_video
from results import save_scores_to_csv, plot_scores
from evaluation import evaluate_selected_clips
from soccernet_labels import load_soccernet_ground_truth


VIDEO_PATH = "data/1_224p.mkv"

SOCCERNET_LABELS_PATH = r"C:\Users\elelw\OneDrive\Desktop\Honours\Semester 1\Biometrics\Biometrics code\SoccerNetData\england_epl\2014-2015\2015-02-22 - 19-15 Southampton 0 - 2 Liverpool\Labels-v2.json"

OUTPUT_DIR = "outputs/clips"

CLIP_LENGTH_SECONDS = 5
MAX_HIGHLIGHT_CLIPS = 10
MIN_GAP_SECONDS = 10


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    cap.release()

    return fps, total_frames, duration


def generate_clip_timestamps(duration, clip_length):
    clips = []
    start = 0

    while start < duration:
        end = min(start + clip_length, duration)
        clips.append((start, end))
        start += clip_length

    return clips


def save_clip(video_path, start_time, end_time, output_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    current_frame = start_frame

    while current_frame < end_frame:
        success, frame = cap.read()

        if not success:
            break

        writer.write(frame)
        current_frame += 1

    cap.release()
    writer.release()


def print_evaluation(name, results, runtime):
    print()
    print(f"- {name} EVALUATION -")

    print(f"Precision: {results['precision']:.2f}")
    print(f"Recall: {results['recall']:.2f}")
    print(f"F-score: {results['f_score']:.2f}")

    print(f"FAR: {results['far']:.4f}")
    print(f"FRR: {results['frr']:.4f}")

    print()

    print(f"True Positives (TP): {results['true_positives']}")
    print(f"False Positives (FP): {results['false_positives']}")
    print(f"False Negatives (FN): {results['false_negatives']}")
    print(f"True Negatives (TN): {results['true_negatives']}")

    print()
    print(f"Runtime: {runtime:.2f} seconds")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    fps, total_frames, duration = get_video_info(VIDEO_PATH)

    print("- Video Information -")
    print(f"FPS: {fps}")
    print(f"Total frames: {int(total_frames)}")
    print(f"Duration: {duration:.2f} seconds")
    print()

    clips = generate_clip_timestamps(
        duration,
        CLIP_LENGTH_SECONDS
    )

    motion_clip_scores = []
    optical_clip_scores = []

    motion_start_time = time.time()

    print("- Processing clips -")

    for index, (start, end) in enumerate(clips):
        clip_number = index + 1

        print(f"Clip {clip_number}: {start:.2f}s to {end:.2f}s")

        motion_score = calculate_motion_score(
            VIDEO_PATH,
            start,
            end
        )

        optical_flow_score = calculate_optical_flow_score(
            VIDEO_PATH,
            start,
            end
        )

        print(f"Frame-difference score: {motion_score:.2f}")
        print(f"Optical-flow score: {optical_flow_score:.4f}")
        print()

        motion_clip_scores.append({
            "clip_number": clip_number,
            "start": start,
            "end": end,
            "score": motion_score
        })

        optical_clip_scores.append({
            "clip_number": clip_number,
            "start": start,
            "end": end,
            "score": optical_flow_score
        })

        output_path = os.path.join(
            OUTPUT_DIR,
            f"clip_{clip_number}.mp4"
        )

        save_clip(
            VIDEO_PATH,
            start,
            end,
            output_path
        )

    motion_runtime = time.time() - motion_start_time

    print("- Selecting pipeline 1 highlights: frame Difference -")

    selected_motion_clips = select_top_clips(
        motion_clip_scores,
        max_clips=MAX_HIGHLIGHT_CLIPS,
        min_gap_seconds=MIN_GAP_SECONDS
    )

    for clip in selected_motion_clips:
        print(
            f"Clip {clip['clip_number']}: "
            f"{clip['start']:.2f}s to {clip['end']:.2f}s "
            f"| Score: {clip['score']:.2f}"
        )

    print()
    print("- Selecting pipeline 2 highlights: optical flow -")

    selected_optical_clips = select_top_clips(
        optical_clip_scores,
        max_clips=MAX_HIGHLIGHT_CLIPS,
        min_gap_seconds=MIN_GAP_SECONDS
    )

    for clip in selected_optical_clips:
        print(
            f"Clip {clip['clip_number']}: "
            f"{clip['start']:.2f}s to {clip['end']:.2f}s "
            f"| Score: {clip['score']:.4f}"
        )

    create_highlight_video(
        VIDEO_PATH,
        selected_motion_clips,
        "outputs/final_highlight_motion.mp4"
    )

    create_highlight_video(
        VIDEO_PATH,
        selected_optical_clips,
        "outputs/final_highlight_optical_flow.mp4"
    )

    save_scores_to_csv(
        motion_clip_scores,
        selected_motion_clips,
        "outputs/clip_scores_motion.csv"
    )

    save_scores_to_csv(
        optical_clip_scores,
        selected_optical_clips,
        "outputs/clip_scores_optical_flow.csv"
    )

    plot_scores(
        motion_clip_scores,
        selected_motion_clips,
        "outputs/motion_scores_graph.png",
        "Pipeline 1: Frame-Difference Motion Scores"
    )

    plot_scores(
        optical_clip_scores,
        selected_optical_clips,
        "outputs/optical_flow_scores_graph.png",
        "Pipeline 2: Optical-Flow Scores"
    )

    if os.path.exists(SOCCERNET_LABELS_PATH):
        ground_truth_events = load_soccernet_ground_truth(
            SOCCERNET_LABELS_PATH
        )

        motion_results = evaluate_selected_clips(
            selected_motion_clips,
            ground_truth_events,
            len(clips)
        )

        optical_results = evaluate_selected_clips(
            selected_optical_clips,
            ground_truth_events,
            len(clips)
        )

        print_evaluation(
            "PIPELINE 1: FRAME DIFFERENCE",
            motion_results,
            motion_runtime
        )

        print_evaluation(
            "PIPELINE 2: OPTICAL FLOW",
            optical_results,
            motion_runtime
        )

        print()
        print("- PIPELINE COMPARISON SUMMARY -")
        print("Pipeline 1 output: outputs/final_highlight_motion.mp4")
        print("Pipeline 2 output: outputs/final_highlight_optical_flow.mp4")

    else:
        print()
        print("No SoccerNet labels found.")
        print("Skipping automatic Precision/Recall/F-score/FAR/FRR evaluation.")
    print()
    print("Done.")
    print("Saved Pipeline 1 highlight to: outputs/final_highlight_motion.mp4")
    print("Saved Pipeline 2 highlight to: outputs/final_highlight_optical_flow.mp4")
    print("Saved Pipeline 1 CSV to: outputs/clip_scores_motion.csv")
    print("Saved Pipeline 2 CSV to: outputs/clip_scores_optical_flow.csv")
    print("Saved graphs to outputs folder.")


if __name__ == "__main__":
    main()