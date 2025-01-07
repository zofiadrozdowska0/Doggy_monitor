import cv2
import numpy as np
import torch
from ultralytics import YOLO

class ProcessStream:
    def __init__(self, model_path, rasa_psa="3", box_iou_thresh=0.55, box_conf_thresh=0.30, kpt_conf_thresh=0.68):
        self.model = YOLO(model_path)
        self.rasa_psa = rasa_psa
        self.box_iou_thresh = box_iou_thresh
        self.box_conf_thresh = box_conf_thresh
        self.kpt_conf_thresh = kpt_conf_thresh
        self.past_emotions = []
        self.pred_boxes = []

    def resize_image(self, frame):
        height, width = frame.shape[:2]
        new_height = (height // 32 + 1) * 32 if height % 32 != 0 else height
        new_width = (width // 32 + 1) * 32 if width % 32 != 0 else width
        resized_image = cv2.resize(frame, (new_width, new_height))
        return resized_image
    

    def decyzja(self, ogon_pozycja, glowa_pozycja, uszy_pozycja, lapy_pozycja, visible_teeth, visible_tongue):
        if ogon_pozycja == 2:
            if lapy_pozycja == 1:  # Zgięte
                return "Happy"
            else:  # Proste
                if visible_tongue:  # Głowa do góry
                    return "Happy"
                else:  # Głowa na dół
                    if visible_teeth:  # Zęby widoczne
                        return "Angry"
                    else:  # Zęby niewidoczne
                        return "Relaxed"
        elif ogon_pozycja == 1:
            if self.rasa_psa == "2" and uszy_pozycja == 1:
                return "Sad"
            else:
                if lapy_pozycja == 1:  # Zgięte
                    if glowa_pozycja == 2:  # Głowa do góry
                        if visible_teeth:
                            return "Angry"
                        else:
                            return "Relaxed"
                    else:  # Głowa na dół
                        if visible_teeth:
                            return "Angry"
                        else:
                            return "Sad"
                else:  # Proste
                    if glowa_pozycja == 2:
                        if visible_teeth:
                            return "Angry"
                        else:
                            return "Relaxed"
                    else:  # Zęby niewidoczne
                        return "Sad"
        elif ogon_pozycja == 3:
            if self.rasa_psa == "2" and uszy_pozycja == 1:
                return "Sad"
            else:
                if lapy_pozycja == 1:  # Zgięte
                    if glowa_pozycja == 2:
                        if visible_teeth:
                            return "Angry"
                        else:
                            return "Happy"
                    else:
                        return "Sad"
                else:  # Proste
                    if visible_teeth:
                        return "Angry"
                    else:
                        return "Relaxed"
        else:
            return "Tail not detected. Can't read emotions."


    def calcuate_emotion(self, angle_lpl, angle_ogon, angle_lpp, angle_glowa,angle_pu,angle_lu, visible_tongue, visible_teeth):
        ogon_pozycja = 0  # 1 - opuszczony, 2 - uniesiony, 3 - wyprostowany
        glowa_pozycja = 0  # 1 - opuszczona, 2 - uniesiona
        uszy_pozycja = 0  # 1 - opuszczone, 2 - uniesione
        lapy_pozycja = 0  # 1 - zgiete, 2 - wyprostowane
        # ogon
        if angle_ogon is not None:
            if 0 < angle_ogon <= 100:
                ogon_pozycja = 2
            elif 100 < angle_ogon <= 240:
                ogon_pozycja = 3
            elif 0 < angle_ogon <= 20:
                ogon_pozycja = 3
            elif angle_ogon > 230:
                ogon_pozycja = 1

        # głowa
        if angle_glowa is not None:
            if angle_glowa <= 10:
                glowa_pozycja = 2
                #print("Głowa uniesiona")
            elif angle_glowa > 10:
                glowa_pozycja = 1
                #print("Głowa opuszczona")


        # łapa przednia lewa i łapa przednia prawa

        if angle_lpl is not None:
            if 140 < angle_lpl <= 180: 
                lapy_pozycja = 2 
                print("Lapy wyprostowane")       
            elif 110 < angle_lpl <= 140: 
                lapy_pozycja = 1
                print("Lapy zgięte")
            elif 80 < angle_lpl <= 110:
                lapy_pozycja = 2
                print("Lapy wyprostowane")

        elif angle_lpp is not None:
            if 140 < angle_lpp <= 180:
                lapy_pozycja = 2 
                print("Lapy wyprostowane")       
            elif 110 < angle_lpp <= 140:
                lapy_pozycja = 1
                print("Lapy zgięte")
            elif 80 < angle_lpp <= 110:
                lapy_pozycja = 2
                print("Lapy wyprostowane")

        # uszy
        if self.rasa_psa == "2":
            if angle_pu is not None:
                if -150 <= angle_pu:
                    uszy_pozycja = 2
                elif angle_pu < -150:
                    uszy_pozycja = 1

            elif angle_lu is not None:
                if -150 <= angle_lu:
                    uszy_pozycja = 2
                elif angle_lu < -150:
                    uszy_pozycja = 1

        return self.decyzja(ogon_pozycja, glowa_pozycja, uszy_pozycja, lapy_pozycja, visible_teeth, visible_tongue)
        #return ogon_pozycja, glowa_pozycja, uszy_pozycja, lapy_pozycja, visible_teeth, visible_tongue


    def calculate_angle(self, p0, p1, p2):
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


    def calculate_intersection_point(self, p1, p2, p3, p4):
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


    # Function to calculate the angle between two lines defined by three points each
    def calculate_intersection_angle(self, p1, p2, p3, p4):
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
    def draw_boxes(self, image, boxes, score=None, color=(0, 255, 0)):
        x1, y1, x2, y2 = boxes
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if score is not None:
            cv2.putText(image, f"Conf: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return image


    def draw_landmarks(self, image, keypoints):
        for x, y, kpt_id in keypoints:
            cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)
            cv2.putText(image, str(int(kpt_id)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return image
    

    def process_frame(self, frame, BOX_IOU_THRESH=0.55, BOX_CONF_THRESH=0.30, KPT_CONF_THRESH=0.68):
        angle_lpl = angle_lpp = angle_ltp = angle_ltl = angle_glowa =angle_pu=angle_lu= angle_pysk = angle_ogon = None
        visible_tongue = False
        visible_teeth = False
        resized_frame = self.resize_image(frame)
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frames_batch_resized = np.transpose(np.expand_dims(frame_rgb, axis=0), (0, 3, 1, 2))
        frames_batch_resized = torch.from_numpy(frames_batch_resized).float() / 255.0

        # Get results from the large model
        results_duzy = self.model_duzy(frames_batch_resized, conf=BOX_CONF_THRESH, iou=BOX_IOU_THRESH)

        result_duzy = results_duzy[0].cpu()


        if len(result_duzy.boxes.xyxy):

            # Get the predicted boxes, conf scores and keypoints.
            pred_box_conf = result_duzy.boxes.conf.numpy()
            max_conf = np.argmax(pred_box_conf)
            boxes = result_duzy.boxes.xyxy.numpy()
            pred_boxes=[boxes[max_conf]]
            # print()
            # pred_kpts_xy = result_duzy.keypoints.xy.numpy()
            # # print(pred_kpts_xy)
            # pred_kpts_conf = result_duzy.keypoints.conf.numpy()
            pred_kpts = result_duzy.keypoints.xy.numpy()
            pred_kpts_xy = [pred_kpts[max_conf]]
            pred_kpt_conf = result_duzy.keypoints.conf.numpy()
            pred_kpts_conf = [pred_kpt_conf[max_conf]]
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
                    angle_lpp = np.abs(self.calculate_angle(p0, p1, p2))
                

                if not np.any(p3 == 0.0) and not np.any(p4 == 0.0) and not np.any(p5 == 0.0):
                    angle_lpl = np.abs(self.calculate_angle(p3, p4, p5))

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
                    angle_glowa = self.calculate_intersection_angle(line1_points[0], line1_points[1], p20,p14)
                elif line1_points and np.all(p15 != 0.0) and np.all(p20 != 0.0):
                    angle_throat = self.calculate_intersection_angle(line1_points[0], line1_points[1], p20,p15)
                    angle_glowa=-2.78+0.06*angle_throat


                if np.all(p14 != 0.0) and np.all(p12 != 0.0) and np.all(p13 != 0.0):
                    print(p12)
                    print(p13)
                    if p12[1]>p13[1]:
                        angle_ogon = self.calculate_angle(p14, p12, p13)
                    else:
                        angle_ogon = 360 - self.calculate_angle(p14, p12, p13)
                elif np.all(p15 != 0.0) and np.all(p12 != 0.0) and np.all(p13 != 0.0):
                    if p12[1]>p13[1]:
                        angle_throat = self.calculate_angle(p15, p12, p13)
                        angle_ogon=18.7+0.98*angle_throat
                    else:
                        angle_throat = 360-self.calculate_angle(p15, p12, p13)
                        angle_ogon = 18.7 + 0.98 * angle_throat
                

                if self.rasa_psa == "2":
                    #prawe ucho
                    if not np.any(p16 == 0.0) and not np.any(p17 == 0.0) and not np.any(p20 == 0.0):
                        angle_pu = self.calculate_angle(p20,p16,p17)

                    #lewe ucho
                    if not np.any(p18 == 0.0) and not np.any(p19 == 0.0) and not np.any(p20 == 0.0):
                        angle_lu = self.calculate_angle(p20,p18,p19)

                p24=filter_kpts[24][:2]
                if not np.any(p24 == 0.0):
                    visible_tongue=True

                p25 = filter_kpts[25][:2]
                p26 = filter_kpts[26][:2]
                if not np.any(p25 == 0.0) or not np.any(p26 == 0.0):
                    visible_teeth = True


            for boxes, score, kpts, confs in zip(pred_boxes, pred_box_conf, pred_kpts_xy, pred_kpts_conf):
                kpts_ids = np.arange(len(kpts))  # Include all keypoints
                filter_kpts = kpts[kpts_ids]
                filter_kpts = np.concatenate([filter_kpts, np.expand_dims(kpts_ids, axis=-1)], axis=-1)
                frame = self.draw_boxes(frame, boxes, score=score, color=(0, 255, 0))
                frame = self.draw_landmarks(frame, filter_kpts)

        wynik = self.calcuate_emotion(angle_lpl, angle_ogon, angle_lpp, angle_glowa,angle_pu,angle_lu, visible_tongue, visible_teeth)
        self.past_emotions.append(wynik)

        if len(self.ast_emotions) > 10:
            self.past_emotions.pop(0)

        self.determine_dominant_emotion(self.past_emotions)

        return frame

    def determine_dominant_emotion(self, past_emotions):
        if len(past_emotions) < 10:
            print("Za mało danych, aby określić dominującą emocję.")
            return None
        
        # Pobierz ostatnie 10 emocji
        recent_emotions = past_emotions[-10:]
        
        dominant_emotion = max(set(recent_emotions), key=recent_emotions.count)
        
        print(f"Dominująca emocja z ostatnich 10 klatek to: {dominant_emotion}")
        return dominant_emotion
    

