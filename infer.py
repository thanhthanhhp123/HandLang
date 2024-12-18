import cv2
import numpy
import os

# Đọc ảnh từ camera rồi ghi lại
class_ = 'O'
image_path = 'data/{}/'.format(class_)
if not os.path.exists(image_path):
    os.makedirs(image_path)

cap = cv2.VideoCapture(1)
cap_x_begin = 0.5
cap_y_end = 0.8
def save_images(frame):
    roi = frame[0:int(cap_y_end*frame.shape[0]), int(cap_x_begin*frame.shape[1]):frame.shape[1]]
    cv2.imwrite(image_path + 'image_{}.png'.format(len(os.listdir(image_path))), roi)
    print('Saved image_{}.png'.format(len(os.listdir(image_path))))


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.rectangle(frame, (int(cap_x_begin*frame.shape[1]), 0), (frame.shape[1], int(cap_y_end*frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    save_images(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

    