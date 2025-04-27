import face_recognition
import cv2
# from tensorflow.keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
print("Current working directory: ", os.getcwd())

webcam_video_stream = cv2.VideoCapture(1)

face_exp_model = model_from_json(open("models/facial_expression_model_structure.json", "r").read())
face_exp_model.load_weights('models/facial_expression_model_weights.h5')
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

all_face_locations = []

while True:
    ret, current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small, model = "cnn")

    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        print("Found face {} at location Top: {}, Left: {}, Bottom: {}, Right: {}".format(index + 1, top_pos, left_pos, bottom_pos, right_pos))
        current_face_image = current_frame_small[top_pos:bottom_pos, left_pos:right_pos]
        #draw rectangle
        cv2.rectangle(current_frame_small, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
        #cv2.imshow("Webcam Video", current_frame_small)

        #blur
        # current_face_image = cv2.GaussianBlur(current_face_image, (99, 99), 30)
        # current_frame_small[top_pos:bottom_pos, left_pos:right_pos] = current_face_image

        current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        current_face_image = cv2.resize(current_face_image, (48, 48))
        img_pixels = image.img_to_array(current_face_image)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        exp_predictions = face_exp_model.predict(img_pixels)
        max_index = np.argmax(exp_predictions[0])
        emotion_label = emotions_label[max_index]

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame_small, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

    cv2.imshow("Webcam Video", current_frame_small)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()