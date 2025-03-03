## AUTOMATIC FACE ZOOM
## SUGGESTED PLACES TO CUT/SPEED UP
## GRAPHIC AT THE END IF TIME

"""
Automatic Face Zoom

"""

from datetime import timedelta
import json
import os
# Get script from YT
with open('transcript.txt', 'r') as file:
    transcript_lines = file.readlines()

transcript_json_lines = []
current_timestamp = None
current_content = []

for line in transcript_lines:
    line = line.strip()
    if not line:  
        continue
    if ':' in line and len(line.split(':')[0]) <= 2:
        if current_timestamp is not None and current_content:
            transcript_json_lines.append({
                "timestamp": current_timestamp,
                "content": ' '.join(current_content).strip()
            })
        current_timestamp = line
        current_content = []
        
    else:
        current_content.append(line)

if current_timestamp is not None and current_content:
    transcript_json_lines.append({
        "timestamp": current_timestamp,
        "content": ' '.join(current_content).strip()
    })

transcript_json = json.dumps(transcript_json_lines, indent=4)



import subprocess

def extract_frame(video_path, output_image_path, timestamp='1'):
    command = [
        'ffmpeg',
        '-ss', timestamp,
        '-i', video_path,
        '-frames:v', '1',
        output_image_path,
        '-y'
    ]
    subprocess.run(command)

video_file = 'video.mp4'  
output_image = 'frame_at_1_second.jpg'  
extract_frame(video_file, output_image)



import cv2
import numpy as np
import requests

def analyze_face_pixels(image_path):

    image = cv2.imread(image_path)


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    faces = face_cascade.detectMultiScale(image_rgb, scaleFactor=1.2, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  
        return {
            "top_left": (x, y),
            "top_right": (x + w, y),
            "bottom_left": (x, y + h),
            "bottom_right": (x + w, y + h),
            "scale": (w, h)
        }
    return None  
face_coordinates = analyze_face_pixels(output_image)

face_coordinates_json = face_coordinates

# print(face_coordinates_json)

def zoom_in_on_face(image_path, face_coordinates, zoom_factor=2):
    image = cv2.imread(image_path)

    if face_coordinates is not None:
        x, y, w, h = face_coordinates["top_left"][0], face_coordinates["top_left"][1], face_coordinates["scale"][0], face_coordinates["scale"][1]

        print(f"Face coordinates: top_left=({x}, {y}), scale=({w}, {h}), full_width={image.shape[1]}, full_height={image.shape[0]}")
        center_x, center_y = x + w // 2, y + h // 2

        new_w, new_h = w * zoom_factor, h * zoom_factor
        new_x = max(center_x - new_w // 2, 0)
        new_y = max(center_y - new_h // 2, 0)

        new_x_end = min(new_x + new_w, image.shape[1])
        new_y_end = min(new_y + new_h, image.shape[0])

        zoomed_face = image[new_y:new_y_end, new_x:new_x_end]

        zoomed_face_path = 'zoomed_face.jpg'
        cv2.imwrite(zoomed_face_path, zoomed_face)

        return zoomed_face_path
    return None

zoomed_face_image_path = zoom_in_on_face(output_image, face_coordinates)



from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
import os
import json

# Load the OpenAI API key from the environment variable

def calculate_heat(transcript_json):
    prompt = """
    You are an expert video analyst tasked with deciding whether to show a face or a
    computer screen based on transcript content. 
    
    Given a JSON transcript with timestamps and text, output a JSON array where each 
    object contains 'timestamp', 'content', and a 'heat' value (0–1) that indicates the 
    likelihood that an active screen action is occurring. 
    
    Use these guidelines: 
    (1) A heat of 0.5 is the threshold where the computer screen is shown. 
    (2) Increase the heat only when an active action is described (for example, 
    'open up outreach' should be 1), but keep it low (around 0) when 
    merely explaining. 
    (3) When transitioning between segments, adjust the heat gradually—limit changes
    to a maximum of 0.2 unless a clear action is present. 
    (4) Do not show either the screen or face for more than 30 consecutive seconds. 
    Output must be valid JSON strictly
    
    For example, "Let's open up outreach" would have a heat score of 1, as outreach is being opened on the screen.
    For example, "welcome you today to learning more about" would have a heat score of 0, as something is just being explained.
    For example, "you've come to expect from the latest" would greatly depend on the heat score of the previous batch of words, and would likely not change more than 0.2 from the previous score
    For example, "well if you might see here trus converts", would have a score of 0.8, as here is likely refferring to something on the computer screen
    For example, "inside of your existing sales tool", may imply that something is being shown, but also may be talking in general. Therefore, this would have little effect
    
    IMPORTANT: Your response must be valid JSON and must strictly follow this format:
    [
        {
            "timestamp": "0:00",
            "content": "hey everyone ainka here one of the",
            "heat": 0
        },
        
    ]

    Do not include any other text in your response, only the JSON array.
    """

    analysis_prompt = prompt + "\n\nAnalyze this transcript:\n" + transcript_json
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0,
    )
    result = response.choices[0].message.content.strip()  # Strip any leading/trailing whitespace
    
    # Attempt to parse the response as JSON
    try:
        heat_scores = json.loads(result)
        return heat_scores
    except json.JSONDecodeError as e:
        print(f"Error parsing OpenAI response as JSON: {e}")
        print(f"Raw response: {result}")
        return []

    except Exception as e:
        print(f"Error in calculate_heat: {e}")
        return []




def calculate_importance(transcript_json):
    prompt = """
    You are an expert video analyst tasked with evaluating the importance of 
    each video segment for trimming purposes. Given a JSON transcript with 
    timestamps and text, output a JSON array where each object contains 'timestamp',
    'content', and an 'importance' value (0–1) that indicates how essential that segment is.
    Segments with an importance score below 0.2 should be cut. 
    Use these guidelines: 
    (1) An importance score of 0.2 is the threshold for retention, anything below should 
    be removed. 
    (2) Increase the importance score only when an active, critical action is described 
    (for example, 'let's open up outreach' should be 1), but keep it low (around 0) for 
    mere explanations.
    (3) When transitioning between segments, adjust the importance gradually—limit changes 
    to a maximum of 0.2 unless a significant action is present. 
    (4) Do not retain more than 30 consecutive seconds of low-importance content. 
    Output must be valid JSON    
    
    For example, "Let's open up outreach" would have a heat score of 1, as outreach is being opened on the screen.
    For example, "welcome you today to learning more about" would have a heat score of 0, as something is just being explained.
    For example, "you've come to expect from the latest" would greatly depend on the heat score of the previous batch of words, and would likely not change more than 0.2 from the previous score
    For example, "well if you might see here trus converts", would have a score of 0.8, as here is likely refferring to something on the computer screen
    For example, "inside of your existing sales tool", may imply that something is being shown, but also may be talking in general. Therefore, this would have little effect
    
    IMPORTANT: Your response must be valid JSON and must strictly follow this format:
    [
        {
            "timestamp": "0:00",
            "content": "hey everyone ainka here one of the",
            "importance": 0
        },
        
    ]

    Do not include any other text in your response, only the JSON array.
    """

    analysis_prompt = prompt + "\n\nAnalyze this transcript:\n" + transcript_json
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0,
    )
    result = response.choices[0].message.content.strip()  # Strip any leading/trailing whitespace
    
    # Attempt to parse the response as JSON
    try:
        heat_scores = json.loads(result)
        return heat_scores
    except json.JSONDecodeError as e:
        print(f"Error parsing OpenAI response as JSON: {e}")
        print(f"Raw response: {result}")
        return []

    except Exception as e:
        print(f"Error in calculate_heat: {e}")
        return []



# Save the following in transcript_heat.txt
# heat_scores = calculate_heat(transcript_json)


# Save teh follwing in transcript_importance.txt
# importance_scores = calculate_importance(transcript_json)
# print(importance_scores)


def write_final_cuts(heat_scores, importance_scores):
    coordinates = []
    for entry in heat_scores:
        importance = next((imp['importance'] for imp in importance_scores if imp['timestamp'] == entry['timestamp']), 0)
        if entry['heat'] >= 0.5:
            coordinates.append({
                "timestamp": entry["timestamp"],
                "cut": "top_left=(0, 0), scale=(640, 360)",
                "importance": importance  
            })
        else:
            coordinates.append({
                "timestamp": entry["timestamp"],
                "cut": "top_left=(40, 299), scale=(30, 30)",
                "importance": importance  
            })
    return coordinates

with open('transcript_heat.txt', 'r') as file:
    transcript_heat = file.read()

with open('transcript_importance.txt', 'r') as file:
    transcript_importance = file.read()

with open('final_cuts.txt', 'w') as file:
    file.write(json.dumps(write_final_cuts(json.loads(transcript_heat), json.loads(transcript_importance)), indent=4))
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import json

# Load the cuts data
with open('final_cuts.txt', 'r') as file:
    cuts_data = json.load(file)

# Prepare the timeline
timestamps = [datetime.strptime(cut['timestamp'], "%H:%M") for cut in cuts_data]
importance_levels = [cut['importance'] for cut in cuts_data]

# Create a color map based on importance
colors = ['red' if importance <= 0.2 else 'yellow' if importance == 0.6 else 'green' for importance in importance_levels]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Smooth the importance levels using a line plot with color based on importance
for i in range(len(timestamps) - 1):
    ax.plot(timestamps[i:i+2], importance_levels[i:i+2], color=colors[i], linewidth=3, alpha=0.8)

# Highlight changes in cut position
for i in range(1, len(cuts_data)):
    if cuts_data[i]['cut'] != cuts_data[i-1]['cut']:
        ax.axvline(x=timestamps[i], color='blue', linestyle='--', linewidth=2, label='Cut Change' if i == 1 else "")
        ax.text(timestamps[i], max(importance_levels) + 0.1, 'Cut Change', color='blue', fontsize=10, ha='center')

# Format the x-axis to show time every minute
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=25)) 
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.xticks(rotation=45)

# Add labels and title
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Importance Level', fontsize=12)
ax.set_title('Timeline', fontsize=14, fontweight='bold')

# Show the plot
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()