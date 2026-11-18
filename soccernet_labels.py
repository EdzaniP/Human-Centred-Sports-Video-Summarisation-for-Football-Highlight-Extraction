import json


def convert_game_time_to_seconds(game_time):
    
    #Converts SoccerNet gameTime format into seconds.
    
    

    half, time_part = game_time.split(" - ")

    minutes, seconds = time_part.split(":")

    total_seconds = (int(minutes) * 60) + int(seconds)

    # Add 45 minutes if second half
    if half.strip() == "2":
        total_seconds += 45 * 60

    return total_seconds


def load_soccernet_ground_truth(labels_path):
    with open(labels_path, "r") as file:
        data = json.load(file)

    events = []

    annotations = data["annotations"]

    for annotation in annotations:
        game_time = annotation["gameTime"]
        label = annotation["label"]

        event_time = convert_game_time_to_seconds(game_time)

        events.append({
            "time": event_time,
            "event": label
        })

    return events