import cv2
import os
import pickle
import face_recognition

imgs_path = 'static/faces'

def encoding_generator(imgs_path):
    
    imglist = os.listdir(imgs_path)
    images = []
    ids = []
    for img in imglist:
        images.append(cv2.imread(os.path.join(imgs_path, img)))
        ids.append(os.path.splitext(img)[0])

    encoded_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        encoded_list.append(encoding)

    return encoded_list, ids

def encode(imgs_path):

    encoded_imgs_with_id = encoding_generator(imgs_path)
    file = open('static/encoded_file.p', 'wb')
    pickle.dump(encoded_imgs_with_id, file)
    file.close()