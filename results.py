import csv
import matplotlib.pyplot as plt


def save_scores_to_csv(clip_scores, selected_clips, output_path):
    selected_numbers = [clip["clip_number"] for clip in selected_clips]

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "clip_number",
            "start_time",
            "end_time",
            "score",
            "selected_for_highlight"
        ])

        for clip in clip_scores:
            writer.writerow([
                clip["clip_number"],
                clip["start"],
                clip["end"],
                clip["score"],
                clip["clip_number"] in selected_numbers
            ])


def plot_scores(clip_scores, selected_clips, output_path, title):
    clip_numbers = [clip["clip_number"] for clip in clip_scores]
    scores = [clip["score"] for clip in clip_scores]

    selected_numbers = [clip["clip_number"] for clip in selected_clips]
    selected_scores = [clip["score"] for clip in selected_clips]

    plt.figure(figsize=(12, 6))
    plt.plot(clip_numbers, scores, marker="o")
    plt.scatter(selected_numbers, selected_scores, s=80)

    plt.title(title)
    plt.xlabel("Clip Number")
    plt.ylabel("Score")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()