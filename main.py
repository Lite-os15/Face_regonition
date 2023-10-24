import cv2
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
sfr.load_encoding_images("Dataset/")


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    face_location, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_location,face_names):
        y1, x1, y2, x2 = face_loc[0],face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame,name,(x1,y2 - 10), cv2.FONT_ITALIC, 1, (0,0,200), 2)
        cv2.rectangle(frame, (x1, y1), (y2, x2), (0, 0, 200), 2)

    # Check if the camera read was successful
    if not ret:
        print("Error reading frame from the camera")
        break

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()