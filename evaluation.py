# Checking if the chosen highlight overlaps with a real SoccerNet event.
# The tolerance allows flexibility because the extracted highlight may not
# perfectly align with the exact timestamp in the annotation.
# tolerance=45 means the system allows a 45-second window around the clip.
def check_overlap(selected_start, selected_end, event_time, tolerance=45):

    # Return True if the event time falls inside the clip range
    # plus the tolerance window.
    return selected_start - tolerance <= event_time <= selected_end + tolerance


# Calculating Precision, Recall, F-score, FAR, and FRR.
# This compares selected highlights with SoccerNet ground-truth events.
def evaluate_selected_clips(selected_clips, ground_truth_events, total_clips):

    # True Positive:
    # A selected highlight clip that overlaps with a real SoccerNet event.
    true_positives = 0

    # Stores matched SoccerNet events so that each real event is only counted once.
    matched_events = set()

    # Loop through all selected highlight clips.
    for clip in selected_clips:

        # Track whether the current selected clip matched a real event.
        matched = False

        # Loop through all real SoccerNet events.
        for index, event in enumerate(ground_truth_events):

            # Get the timestamp of the real football event.
            event_time = event["time"]

            # Check if the selected clip overlaps with the real event.
            if check_overlap(clip["start"], clip["end"], event_time):

                # This selected clip is a correct detection.
                matched = True

                # Store the matched event index.
                matched_events.add(index)

                # Stop checking this selected clip after the first match.
                break

        # If the clip matched any event, count it as a true positive.
        if matched:
            true_positives += 1

    # False Positive:
    # A selected highlight clip that does NOT match any SoccerNet event.
    false_positives = len(selected_clips) - true_positives

    # False Negative:
    # A real SoccerNet event that was NOT captured by any selected highlight.
    false_negatives = len(ground_truth_events) - len(matched_events)

    # True Negative:
    # A non-event clip that was correctly NOT selected as a highlight.
    true_negatives = (
        total_clips
        - true_positives
        - false_positives
        - false_negatives
    )

    # Prevent negative true negatives in case the annotation count is larger
    # than the number of generated clips.
    if true_negatives < 0:
        true_negatives = 0

    # Precision:
    # correct selected highlights / total selected highlights.
    # High precision means selected highlights are mostly relevant.
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )

    # Recall:
    # captured real events / total real events.
    # High recall means the summary captures many annotated events.
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    # F-score:
    # Harmonic mean between precision and recall.
    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)

    # FAR: False Acceptance Rate.
    # false positives / all actual non-events.
    # In this project, this means:
    # rate of non-event clips incorrectly selected as highlights.
    far = (
        false_positives / (false_positives + true_negatives)
        if (false_positives + true_negatives) > 0
        else 0
    )

    # FRR: False Rejection Rate.
    # false negatives / all actual real events.
    # In this project, this means:
    # rate of real football events missed by the summarisation system.
    frr = (
        false_negatives / (false_negatives + true_positives)
        if (false_negatives + true_positives) > 0
        else 0
    )

    # Returning all evaluation results.
    return {
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "far": far,
        "frr": frr,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives
    }