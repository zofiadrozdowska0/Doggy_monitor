import cv2
import numpy as np
from ultralytics import YOLO
import torch
import math

# Load models
model_duzy_path = './models/model_3.pt'
model_duzy = YOLO(model_duzy_path)  # Use GPU if available
input_path = './psikod/piesel.mp4'
output_path = 'piesel_framed.mp4'

rasa_psa = "3"

def read_text_file(file_path):
    """Reads the text file and returns a list of rows."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def process_text_file(text_lines):
    """Processes the text file by taking every 4 rows and choosing the highest percentage plus the relaxed percentage."""
    percentages = []
    for i in range(0, len(text_lines), 4):
        # Take every group of 4 rows
        group = text_lines[i:i + 4]
        # Extract percentages from each line, handling 'nan' values
        values = []
        for line in group:
            try:
                # Extract the percentage as a float, stripping the '%' and handling 'nan'
                percentage_str = line.split()[-2].strip('%')
                if percentage_str.lower() == 'nan':
                    values.append(0)  # If it's 'nan', consider it as 0
                else:
                    percentage = float(percentage_str)
                    values.append(percentage)
            except ValueError:
                values.append(0)  # In case there's a line with non-numeric value

        if values:
            # Get the highest percentage from the first three rows (Happy, Angry, Sad)
            max_value = max(values[:4])  # Take the max from first 3 rows (Happy, Angry, Sad)
            relaxed_value = values[3]  # Take the relaxed value from the 4th row (Relaxed)
            total_percentage = max_value  # Add Relaxed percentage
            row_number = values.index(max_value) + 1  # Row number of the highest percentage from Happy, Angry, Sad
            percentages.append((total_percentage, row_number))
        else:
            percentages.append((None, None))  # In case of no valid percentages
    return percentages


def get_emotion_from_row(row_number):
    """Maps the row number to the corresponding emotion."""
    emotion_map = {
        1: "Happy",
        2: "Angry",
        3: "Sad",
        4: "Relaxed"
    }
    return emotion_map.get(row_number, "Unknown")  # Default to "Unknown" if no match


def display_video_with_percentage(video_path, percentages):
    """Displays the video and shows every 10th frame with its corresponding percentage."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    percentage_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every 10th frame
        if frame_count % 10 == 0:
            if percentage_index < len(percentages):
                total_percentage, row_number = percentages[percentage_index]

                # Get the emotion corresponding to the row number
                emotion = get_emotion_from_row(row_number)

                # Check if the total_percentage is 0, and if so, display "Emotion unknown"
                if total_percentage == 0:
                    total_percentage_text = "Emotion unknown"
                    print(f"Frame {frame_count}, Emotion: Unknown")
                else:
                    total_percentage = round(total_percentage, 2)
                    total_percentage_text = f"{emotion} with {total_percentage}% certainty on last 10 frames "
                    print(f"Frame {frame_count}, Total Percentage: {total_percentage}% ({emotion})")

                # Add text on the frame
                cv2.putText(frame, total_percentage_text, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the frame with percentage
            cv2.imshow('Video with Percentage', frame)

            # Wait for the user to press 'n' for next frame or 'q' to quit
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # Press 'q' to quit
                break
            elif key == ord('n'):  # Press 'n' to show next frame
                percentage_index += 1  # Move to the next frame

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def video_processing_complete():
    print("Video processing is complete.")


happy = 0
relaxed = 0
sad = 0
angry = 0
i = 0

with open("results.txt", "w") as file:
    file.write("")

def calcuate_emotion(angle_lpl, angle_ogon, angle_lpp, angle_ltp, angle_ltl, angle_glowa,angle_pu,angle_lu,angle_pysk,visible_tongue,visible_teeth):
    global happy, relaxed, sad, angry
    global i

    # ogon
    if angle_ogon is not None:
        if 0 < angle_ogon < 90:
            happy += np.sqrt(2 / np.pi * np.arctan(6 * (np.deg2rad(angle_ogon) + np.pi / 18)))*1.7
            h_sr += np.sqrt(2 / np.pi * np.arctan(6 * (np.deg2rad(angle_ogon) + np.pi / 18))) * 1.7
            if visible_teeth:
                angry += np.sqrt(2 / np.pi * np.arctan(6 * (np.deg2rad(angle_ogon) + np.pi / 18)))*0.8
                a_sr += np.sqrt(2 / np.pi * np.arctan(6 * (np.deg2rad(angle_ogon) + np.pi / 18))) * 0.8
        if 180 < angle_ogon < 230:
            normal += -2.7 * (np.deg2rad(angle_ogon) - np.pi * 17 / 18) * (np.deg2rad(angle_ogon) - 24 * np.pi / 18)
            n_sr += -2.7 * (np.deg2rad(angle_ogon) - np.pi * 17 / 18) * (np.deg2rad(angle_ogon) - 24 * np.pi / 18)
        if 0 < angle_ogon < 20:
            normal += -33 * (np.deg2rad(angle_ogon) - np.pi * 2 / 18) * (np.deg2rad(angle_ogon))
            n_sr += -33 * (np.deg2rad(angle_ogon) - np.pi * 2 / 18) * (np.deg2rad(angle_ogon))
        if angle_ogon > 210:
            sad += np.sqrt(2 / np.pi * np.arctan(3 * np.deg2rad(angle_ogon) - 21 * np.pi / 18))
            s_sr += np.sqrt(2 / np.pi * np.arctan(3 * np.deg2rad(angle_ogon) - 21 * np.pi / 18))

    # głowa
    if angle_glowa is not None:
        if angle_glowa < -10:
            happy += np.sqrt(2 / np.pi * np.arctan(-8 * (np.deg2rad(angle_glowa) + np.pi / 36)))*1.7
            h_sr += np.sqrt(2 / np.pi * np.arctan(-8 * (np.deg2rad(angle_glowa) + np.pi / 36))) * 1.7
            angry += np.sqrt(2 / np.pi * np.arctan(-4 * (np.deg2rad(angle_glowa) + np.pi / 36)))*0.6
            a_sr += np.sqrt(2 / np.pi * np.arctan(-4 * (np.deg2rad(angle_glowa) + np.pi / 36))) * 0.6
        if -10 < angle_glowa < 10:
            normal += -30 * (np.deg2rad(angle_glowa) - np.pi / 18) * (np.deg2rad(angle_glowa) + np.pi / 18)
            n_sr += -30 * (np.deg2rad(angle_glowa) - np.pi / 18) * (np.deg2rad(angle_glowa) + np.pi / 18)
        if angle_glowa > 10:
            sad += np.sqrt(2 / np.pi * np.arctan(4 * (np.deg2rad(angle_glowa) - np.pi / 36)))
            s_sr += np.sqrt(2 / np.pi * np.arctan(4 * (np.deg2rad(angle_glowa) - np.pi / 36)))

    if angle_pysk is not None:
        if 40 < angle_pysk < 60 :
            happy += np.sqrt(2 / np.pi * np.arctan(15 * (np.deg2rad(angle_pysk) - 4 * np.pi / 18)))
            h_sr += np.sqrt(2 / np.pi * np.arctan(15 * (np.deg2rad(angle_pysk) - 4 * np.pi / 18)))
        if 30 < angle_pysk < 40:
            normal += -130 * (np.deg2rad(angle_pysk) - np.pi * 3 / 18) * (np.deg2rad(angle_pysk) - 4 * np.pi / 18)
            n_sr += -130 * (np.deg2rad(angle_pysk) - np.pi * 3 / 18) * (np.deg2rad(angle_pysk) - 4 * np.pi / 18)
        if 20 < angle_pysk < 30:
            angry += np.sqrt(2 / np.pi * np.arctan(15 * (np.deg2rad(angle_pysk) - 2 * np.pi / 18)))*0.5
            a_sr += np.sqrt(2 / np.pi * np.arctan(15 * (np.deg2rad(angle_pysk) - 2 * np.pi / 18))) * 0.5
            if visible_teeth:
                angry += np.sqrt(2 / np.pi * np.arctan(15 * (np.deg2rad(angle_pysk) - 2 * np.pi / 18)))*2
                a_sr += np.sqrt(2 / np.pi * np.arctan(15 * (np.deg2rad(angle_pysk) - 2 * np.pi / 18))) * 2
        if 10 < angle_pysk < 20:
            sad += np.sqrt(2 / np.pi * np.arctan(-15 * (np.deg2rad(angle_pysk) - 2 * np.pi / 18)))
            s_sr += np.sqrt(2 / np.pi * np.arctan(-15 * (np.deg2rad(angle_pysk) - 2 * np.pi / 18)))

    # łapa przednia lewa i łapa przednia prawa
    if angle_lpl is not None and angle_lpp is not None:
        if 140 < angle_lpl < 180:
            angry += np.sqrt(2 / np.pi * np.arctan(6 * (np.deg2rad(angle_lpl) - 14 * np.pi / 18)))
            a_sr += np.sqrt(2 / np.pi * np.arctan(6 * (np.deg2rad(angle_lpl) - 14 * np.pi / 18)))
            if not visible_teeth:
                normal += -8 * (np.deg2rad(angle_lpl) - 14 * np.pi / 18) * (np.deg2rad(angle_lpl) - 18 * np.pi / 18)
                n_sr += -8 * (np.deg2rad(angle_lpl) - 14 * np.pi / 18) * (np.deg2rad(angle_lpl) - 18 * np.pi / 18)
        elif 140 < angle_lpp < 180:
            angry += np.sqrt(2 / np.pi * np.arctan(6 * (np.deg2rad(angle_lpp) - 14 * np.pi / 18)))
            a_sr += np.sqrt(2 / np.pi * np.arctan(6 * (np.deg2rad(angle_lpp) - 14 * np.pi / 18)))
            if not visible_teeth:
                normal += -8 * (np.deg2rad(angle_lpp) - 14 * np.pi / 18) * (np.deg2rad(angle_lpp) - 18 * np.pi / 18)
                n_sr += -8 * (np.deg2rad(angle_lpp) - 14 * np.pi / 18) * (np.deg2rad(angle_lpp) - 18 * np.pi / 18)
        if 110 < angle_lpl < 150:
            sad += np.sqrt(2 / np.pi * np.arctan(-8 * (np.deg2rad(angle_lpl) - 15 * np.pi / 18)))
            s_sr += np.sqrt(2 / np.pi * np.arctan(-8 * (np.deg2rad(angle_lpl) - 15 * np.pi / 18)))
        elif 110 < angle_lpp < 150:
            sad += np.sqrt(2 / np.pi * np.arctan(-8 * (np.deg2rad(angle_lpp) - 15 * np.pi / 18)))
            s_sr += np.sqrt(2 / np.pi * np.arctan(-8 * (np.deg2rad(angle_lpp) - 15 * np.pi / 18)))
        if 80 < angle_lpl < 120 or 80 < angle_lpp < 120:
            happy += np.sqrt(2 / np.pi * np.arctan(-6 * (np.deg2rad(angle_lpl) - 12 * np.pi / 18)))
            h_sr += np.sqrt(2 / np.pi * np.arctan(-6 * (np.deg2rad(angle_lpl) - 12 * np.pi / 18)))


    # uszy
    if rasa_psa == "2":
        if angle_pu is not None:
            if -140 < angle_pu < -100:
                happy += np.sqrt(2 / np.pi * np.arctan(6 * (np.deg2rad(angle_pu) + np.pi / 18))*0.4)
                h_sr += np.sqrt(2 / np.pi * np.arctan(6 * (np.deg2rad(angle_pu) + np.pi / 18))) * 0.4
            if -100 < angle_pu:
                angry += -2.7 * (np.deg2rad(angle_pu) - np.pi * 17 / 18) * (np.deg2rad(angle_pu) - 24 * np.pi / 18)*0.4
                a_sr += -2.7 * (np.deg2rad(angle_pu) - np.pi * 17 / 18) * (np.deg2rad(angle_pu) - 24 * np.pi / 18) * 0.4
            if -150 < angle_pu < -140:
                normal += -33 * (np.deg2rad(angle_pu) - np.pi * 2 / 18) * (np.deg2rad(angle_pu))*0.4
                n_sr += -33 * (np.deg2rad(angle_pu) - np.pi * 2 / 18) * (np.deg2rad(angle_pu)) * 0.4
            if angle_pu < -150:
                sad += np.sqrt(2 / np.pi * np.arctan(3 * np.deg2rad(angle_pu) - 21 * np.pi / 18))*0.4
                s_sr += np.sqrt(2 / np.pi * np.arctan(3 * np.deg2rad(angle_pu) - 21 * np.pi / 18)) * 0.4

        elif angle_lu is not None:
            if -140 < angle_lu < -100:
                happy += np.sqrt(2 / np.pi * np.arctan(6 * (np.deg2rad(angle_lu) + np.pi / 18))*0.4)
                h_sr += np.sqrt(2 / np.pi * np.arctan(6 * (np.deg2rad(angle_lu) + np.pi / 18))) * 0.4
            if -100 < angle_lu:
                angry += -2.7 * (np.deg2rad(angle_lu) - np.pi * 17 / 18) * (np.deg2rad(angle_lu) - 24 * np.pi / 18)*0.4
                a_sr += -2.7 * (np.deg2rad(angle_lu) - np.pi * 17 / 18) * (np.deg2rad(angle_lu) - 24 * np.pi / 18) * 0.4
            if -150 < angle_lu < -140:
                normal += -33 * (np.deg2rad(angle_lu) - np.pi * 2 / 18) * (np.deg2rad(angle_lu))*0.4
                n_sr += -33 * (np.deg2rad(angle_lu) - np.pi * 2 / 18) * (np.deg2rad(angle_lu)) * 0.4
            if angle_lu < -150:
                sad += np.sqrt(2 / np.pi * np.arctan(3 * np.deg2rad(angle_lu) - 21 * np.pi / 18))*0.4
                s_sr += np.sqrt(2 / np.pi * np.arctan(3 * np.deg2rad(angle_lu) - 21 * np.pi / 18)) * 0.4


    suma = a_sr + h_sr + n_sr + s_sr
    if suma == 0:
        suma = np.inf
    with open("results.txt", "a") as file:
        file.write(f"happy sr: {h_sr / suma*100} %\n")
        file.write(f"angry sr: {a_sr / suma*100} %\n")
        file.write(f"sad sr: {s_sr / suma*100} %\n")
        file.write(f"relaxed sr: {n_sr / suma*100} %\n")

    i += 1
    if i == 10:
        suma = a_sr+h_sr+n_sr+s_sr
        if suma==0:
            suma=np.inf
        with open("results.txt", "a") as file:
            file.write(f"happy sr: {h_sr / suma*100} %\n")
            file.write(f"angry sr: {a_sr / suma*100} %\n")
            file.write(f"sad sr: {s_sr / suma*100} %\n")
            file.write(f"relaxed sr: {n_sr / suma*100} %\n")

        print("happy sr: ", h_sr/suma*100, " %", "\n")
        print("angry sr: ", a_sr/suma*100, " %", "\n")
        print("sad sr: ", s_sr/suma*100, " %",  "\n")
        print("relaxed sr: ", n_sr/suma*100, " %", "\n")
        a_sr = 0
        h_sr = 0
        n_sr = 0
        s_sr = 0
        i = 0

    print(suma)
    # print("happy: ", happy, "\n")
    # print("angry: ", angry, "\n")
    # print("sad: ", sad, "\n")
    # print("relaxed: ", normal, "\n")


def calculate_angle(p0, p1, p2):
    a = np.array(p0)
    b = np.array(p1)
    c = np.array(p2)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # Determine the direction of the angle using the cross product
    cross_product = np.cross(ba, bc)
    if cross_product < 0:
        angle = -angle

    angle_degrees = np.degrees(angle)

    # Adjust angle to be within -180 to 180
    if angle_degrees > 180:
        angle_degrees -= 360
    elif angle_degrees < -180:
        angle_degrees += 360

    return angle_degrees

# Function to calculate the intersection point of two lines


def calculate_intersection_point(p1, p2, p3, p4):
    # Line AB represented as a1x + b1y = c1
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]

    # Line CD represented as a2x + b2y = c2
    a2 = p4[1] - p3[1]
    b2 = p3[0] - p4[0]
    c2 = a2 * p3[0] + b2 * p3[1]

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        # The lines are parallel. This is simplified
        return None
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return np.array([x, y])


# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Function to calculate the angle between two lines defined by three points each
def calculate_intersection_angle(p1, p2, p3, p4):
    # Define vectors
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p4) - np.array(p3)

    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Calculate the dot product and then the angle
    dot_product = np.dot(v1, v2)
    angle = np.arccos(dot_product)
    return np.degrees(angle)


# Function to draw bounding boxes
def draw_boxes(image, boxes, score=None, color=(0, 255, 0)):
    x1, y1, x2, y2 = boxes
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    if score is not None:
        cv2.putText(image, f"Conf: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

# Function to draw keypoints


def draw_landmarks(image, keypoints):
    for x, y, kpt_id in keypoints:
        cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)
        cv2.putText(image, str(int(kpt_id)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image


# Function to resize the image to be divisible by 32
def resize_image(image):
    height, width = image.shape[:2]
    new_height = (height // 32 + 1) * 32 if height % 32 != 0 else height
    new_width = (width // 32 + 1) * 32 if width % 32 != 0 else width
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


# Function to process a single frame
def process_frame(frame, frame_index, BOX_IOU_THRESH=0.55, BOX_CONF_THRESH=0.30, KPT_CONF_THRESH=0.68):
    angle_lpl = angle_lpp = angle_ltp = angle_ltl = angle_glowa =angle_pu=angle_lu= angle_pysk = angle_ogon = None
    visible_tongue = False
    visible_teeth = False
    resized_frame = resize_image(frame)
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    frames_batch_resized = np.transpose(np.expand_dims(frame_rgb, axis=0), (0, 3, 1, 2))
    frames_batch_resized = torch.from_numpy(frames_batch_resized).float() / 255.0

    # Get results from the large model
    results_duzy = model_duzy(frames_batch_resized, conf=BOX_CONF_THRESH, iou=BOX_IOU_THRESH)

    result_duzy = results_duzy[0].cpu()



    if len(result_duzy.boxes.xyxy):

        # Get the predicted boxes, conf scores and keypoints.
        pred_boxes = result_duzy.boxes.xyxy.numpy()
        pred_box_conf = result_duzy.boxes.conf.numpy()
        pred_kpts_xy = result_duzy.keypoints.xy.numpy()
        pred_kpts_conf = result_duzy.keypoints.conf.numpy()

        # Collect keypoints data
        for kpts, confs in zip(pred_kpts_xy, pred_kpts_conf):
            kpts_ids = np.arange(len(kpts))  # Include all keypoints
            filter_kpts = kpts[kpts_ids]
            filter_kpts = np.concatenate([filter_kpts, np.expand_dims(kpts_ids, axis=-1)], axis=-1)

            p0 = filter_kpts[0][:2]
            p1 = filter_kpts[1][:2]
            p2 = filter_kpts[2][:2]
            p6 = filter_kpts[6][:2]
            p7 = filter_kpts[7][:2]
            p8 = filter_kpts[8][:2]
            p9 = filter_kpts[9][:2]
            p10 = filter_kpts[10][:2]
            p11 = filter_kpts[11][:2]
            p12 = filter_kpts[12][:2]
            p3 = filter_kpts[3][:2]
            p4 = filter_kpts[4][:2]
            p5 = filter_kpts[5][:2]
            p2 = filter_kpts[2][:2]
            p5 = filter_kpts[5][:2]
            p8 = filter_kpts[8][:2]
            p11 = filter_kpts[11][:2]
            p13 = filter_kpts[13][:2]
            p16 = filter_kpts[16][:2]
            p17 = filter_kpts[17][:2]


            #OBLICZANIE KATOW NOG
            # prawa przednia lapa
            if not np.any(p0 == 0.0) and not np.any(p1 == 0.0) and not np.any(p2 == 0.0):
                angle_lpp = np.abs(calculate_angle(p0, p1, p2))
                print(f"Frame {frame_index}: Prawa przednia łapa: {angle_lpp:.2f} degrees")
            # prawa tylna lapa
            if not np.any(p6 == 0.0) and not np.any(p7 == 0.0) and not np.any(p8 == 0.0):
                angle_ltp = np.abs(calculate_angle(p6, p7, p8))
                print(f"Frame {frame_index}: Prawa tylna łapa: {angle_ltp:.2f} degrees")
            # lewa tylna lapa
            if not np.any(p9 == 0.0) and not np.any(p10 == 0.0) and not np.any(p11 == 0.0):
                angle_ltl = np.abs(calculate_angle(p9, p10, p11))
                print(f"Frame {frame_index}: Lewa tylna łapa: {angle_ltl:.2f} degrees")
            # lewa przednia lapa
            if not np.any(p3 == 0.0) and not np.any(p4 == 0.0) and not np.any(p5 == 0.0):
                angle_lpl = np.abs(calculate_angle(p3, p4, p5))
                print(f"Frame {frame_index}: Lewa przednia łapa: {angle_lpl:.2f} degrees")

            # kark i gardło
            p14 = filter_kpts[14][:2]
            p15 = filter_kpts[15][:2]

            # prawy kącik i lewy kącik
            p22 = filter_kpts[22][:2]
            p23 = filter_kpts[23][:2]
            # nos i dolna warga
            p20 = filter_kpts[20][:2]
            p21 = filter_kpts[21][:2]

            # prawe ucho
            p16 = filter_kpts[16][:2]
            p17 = filter_kpts[17][:2]

            #lewe ucho
            p18 = filter_kpts[18][:2]
            p19 = filter_kpts[19][:2]

            # Create line from available points
            line1_points = []
            if np.all(p0 != 0.0) and np.all(p6 != 0.0):
                line1_points = [p0, p6]
            elif np.all(p3 != 0.0) and np.all(p9 != 0.0):
                line1_points = [p3, p9]
            elif np.all(p12 != 0.0) and np.all(p14 != 0.0):
                line1_points = [p12, p14]

            if line1_points and np.all(p14 != 0.0) and np.all(p20 != 0.0):
                angle_glowa = calculate_intersection_angle(line1_points[0], line1_points[1], p20,p14)
                print(f"Frame {frame_index}: Glowa: {angle_glowa:.2f} degrees")
            elif line1_points and np.all(p15 != 0.0) and np.all(p20 != 0.0):
                angle_throat = calculate_intersection_angle(line1_points[0], line1_points[1], p20,p15)
                angle_glowa=-2.78+0.06*angle_throat
                print(f"Frame {frame_index}: Glowa: {angle_glowa:.2f} degrees")


            if np.all(p14 != 0.0) and np.all(p12 != 0.0) and np.all(p13 != 0.0):
                angle_ogon = 180-calculate_intersection_angle(p14, p12, p12, p13)
                print(f"Frame {frame_index}: Ogon: {angle_ogon:.2f} degrees")
            elif np.all(p15 != 0.0) and np.all(p12 != 0.0) and np.all(p13 != 0.0):
                angle_throat = 180-calculate_intersection_angle(p15, p12, p12, p13)
                angle_ogon=18.7+0.98*angle_throat
                print(f"Frame {frame_index}: Ogon: {angle_ogon:.2f} degrees")
            

            if rasa_psa == "2":
                #prawe ucho
                if not np.any(p16 == 0.0) and not np.any(p17 == 0.0) and not np.any(p20 == 0.0):
                    angle_pu = calculate_angle(p20,p16,p17)
                    print(f"Frame {frame_index}: Prawe ucho: {angle_pu:.2f} degrees")

                #lewe ucho
                if not np.any(p18 == 0.0) and not np.any(p19 == 0.0) and not np.any(p20 == 0.0):
                    angle_lu = calculate_angle(p20,p18,p19)
                    print(f"Frame {frame_index}: Lewe ucho: {angle_lu:.2f} degrees")

            #Otwarcie pyska
            if not np.any(p22 == 0.0) and not np.any(p20 == 0.0) and not np.any(p21 == 0.0):
                angle_pysk = calculate_angle(p20,p22,p21)
                print(f"Frame {frame_index}: Pysk: {angle_pysk:.2f} degrees")
            elif not np.any(p23 == 0.0) and not np.any(p20 == 0.0) and not np.any(p21 == 0.0):
                angle_pysk = calculate_angle(p20, p23, p21)
                print(f"Frame {frame_index}: Pysk: {angle_pysk:.2f} degrees")

            #jezyk i zeby
            p24=filter_kpts[24][:2]
            if not np.any(p24 == 0.0):
                visible_tongue=True


            p25 = filter_kpts[25][:2]
            if not np.any(p25 == 0.0):
                visible_teeth = True

            p26 = filter_kpts[26][:2]
            if not np.any(p26 == 0.0):
                visible_teeth= True



        # Draw predicted bounding boxes, conf scores and keypoints on image.
        for boxes, score, kpts, confs in zip(pred_boxes, pred_box_conf, pred_kpts_xy, pred_kpts_conf):
            kpts_ids = np.arange(len(kpts))  # Include all keypoints
            filter_kpts = kpts[kpts_ids]
            filter_kpts = np.concatenate([filter_kpts, np.expand_dims(kpts_ids, axis=-1)], axis=-1)
            frame = draw_boxes(frame, boxes, score=score, color=(0, 255, 0))
            frame = draw_landmarks(frame, filter_kpts)

    calcuate_emotion(angle_lpl, angle_ogon, angle_lpp, angle_ltp, angle_ltl, angle_glowa, angle_pu, angle_lu, angle_pysk, visible_tongue, visible_teeth)

    return frame


# Function to process video
def process_video(input_path, output_path, callback, batch_size=8):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    frames = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames.append((frame_index, frame))
        frame_index += 1

        if len(frames) == batch_size:
            # Process batch of frames
            for i in range(batch_size):
                frames[i] = (frames[i][0], process_frame(frames[i][1], frames[i][0]))
            for processed_frame in frames:
                out.write(processed_frame[1])
            frames = []

    # Process remaining frames
    if frames:
        for i in range(len(frames)):
            frames[i] = (frames[i][0], process_frame(frames[i][1], frames[i][0]))
        for processed_frame in frames:
            out.write(processed_frame[1])

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    callback()
#process_video(input_path, output_path, video_processing_complete, batch_size=8)


def apply_percentage_to_image(image_path, percentages):
    """Applies the highest percentage value on the given image."""
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found!")
        return

    # Assuming there's only one percentage value (for the first group of 4 rows)
    if percentages:
        total_percentage, row_number = percentages[0]

        # Get the emotion corresponding to the row number
        emotion = get_emotion_from_row(row_number)

        # Check if the total_percentage is 0, and if so, display "Emotion unknown"
        if total_percentage == 0:
            total_percentage_text = "Emotion unknown"
        else:
            # Round the total percentage to 2 decimal places
            total_percentage = round(total_percentage, 2)
            total_percentage_text = f"{total_percentage}% ({emotion})"

        # Add text on the image with a smaller font
        cv2.putText(image, total_percentage_text, (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the image with percentage
        cv2.imshow('Image with Percentage', image)

        # Wait for the user to press any key to close the image window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image
def main(img_path, text_file_path):
    # Step 1: Read the text file
    text_lines = read_text_file(text_file_path)

    # Step 2: Process the text file and get the highest percentages
    percentages = process_text_file(text_lines)

    # Step 3: Display the video with every 10th frame and the corresponding percentage
    display_video_with_percentage(img_path, percentages)

def rysiowanie(model, img):
    if img is None:
        raise ValueError("Nie można wczytać obrazu. Sprawdź ścieżkę do pliku.")
    process_frame(img, 0)
    results = model(img)

    print("Kluczowe punkty dla psa:")

    if len(results) > 0 and hasattr(results[0], 'keypoints'):
        for i, keypoint_set in enumerate(results[0].keypoints.data):
            print(f"\nDetekcja {i + 1}:")

            for j, keypoint in enumerate(keypoint_set):
                x, y, confidence = keypoint
                print(f"Punkt {j + 1}: X = {x.item():.2f}, Y = {y.item():.2f}, Confidence = {confidence.item():.2f}")
    else:
        print("Nie wykryto punktów kluczowych dla obiektów na obrazie.")

    for result in results:
        result.show()
        result.save(filename='wyzel_framed.jpg')
        main('wyzel_framed.jpg', text_file_path)


#model = YOLO('./models/model_2.pt')
#img_path = 'aa.jfif'
#img = cv2.imread(img_path)
video_path = 'piesel_framed.mp4'  # Path to the MP4 video file
text_file_path = 'results.txt'  # Path to the text file

#rysiowanie(model, img)
# process_frame(img, 0)
# main(img_path, text_file_path)
process_video(input_path, output_path, video_processing_complete)
main(video_path,text_file_path)