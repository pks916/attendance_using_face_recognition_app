import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
import pickle
import pandas as pd
import face_recognition
import imutils
from encoding_generator import encode
import requests

imgs_path = 'static/faces'

app = Flask(__name__)

def datetoday_numerical():
    return date.today().strftime("%d_%m_%y")
def datetoday_formal():
    return date.today().strftime("%d-%B-%Y")

def initialize():
    if not os.path.isdir('attendances'):
        os.makedirs('attendances')
    if not os.path.isdir(imgs_path):
        os.makedirs(imgs_path)

    if f'attendance-{datetoday_numerical()}.csv' not in os.listdir('attendances'):
        with open(f'attendances/attendance-{datetoday_numerical()}.csv','w') as f:
            f.write('Name,Roll,Time')

def total_entries():
    return len(os.listdir(imgs_path))

def identify_face(facearray):
    return face_recognition(facearray)

def extract_attendance():

    df = pd.read_csv(f'attendances/attendance-{datetoday_numerical()}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    strength = len(df)
    return names, rolls, times, strength

def add_attendance(name):

    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'attendances/attendance-{datetoday_numerical()}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'attendances/attendance-{datetoday_numerical()}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')


@app.route('/')
def home():
    initialize()
    names,rolls,times,strength = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,strength=strength,totalreg=total_entries(),datetoday2=datetoday_formal()) 

@app.route('/stop_attendance', methods=['POST'])
def stop_attendance():
    names,rolls,times,strength = extract_attendance() 
    return render_template('home.html',names=names,rolls=rolls,times=times,strength=strength,totalreg=total_entries(),datetoday2=datetoday_formal(),mess = 'Attendance Stopped')

@app.route('/start',methods=['GET'])
def start():

    if 'encoded_file.p' not in os.listdir('static'):
        return render_template('home.html',totalreg=total_entries(),datetoday2=datetoday_formal(),mess='Please generate encodings of the faces.') 
    
    encoded_imgs, ids = pickle.load(open('static/encoded_file.p','rb'))

    while True:

        img_resp = requests.get(url) 
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
        img = cv2.imdecode(img_arr, -1) 
        frame = imutils.resize(img, width=720, height=640)
        
        faceCurFrame = face_recognition.face_locations(frame)
        encodeCurFrame = face_recognition.face_encodings(frame, faceCurFrame)

        for encodedface, faceloc in zip(encodeCurFrame, faceCurFrame):

            matches = face_recognition.compare_faces(encoded_imgs, encodedface)
            faceDis = face_recognition.face_distance(encoded_imgs, encodedface)
            matchindex = np.argmin(faceDis)

            if matches[matchindex]:
                name = ids[matchindex]
                y1, x2, y2, x1 = faceloc
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                add_attendance(name)

        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)==27:
            break

    # cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=total_entries(),datetoday2=datetoday_formal()) 

@app.route('/add',methods=['GET','POST'])
def add():

    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    encoded_imgs, _ = pickle.load(open('static/encoded_file.p','rb'))
    encoded_imgs = np.array(encoded_imgs)
    
    while True:

        img_resp = requests.get(url) 
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
        img = cv2.imdecode(img_arr, -1) 
        frame = imutils.resize(img, width=720, height=640)

        faceCurFrame = face_recognition.face_locations(frame)

        if len(faceCurFrame)>1:
            cv2.imshow('Adding new User', frame)
            print('One face at a time ...')
            if cv2.waitKey(1)==27:
                break
            continue

        if len(faceCurFrame)==0:
            cv2.imshow('Adding new User', frame)
            print('No face detected ...')
            if cv2.waitKey(1)==27:
                break
            continue
        
        encodeCurFrame = face_recognition.face_encodings(frame, faceCurFrame)
        encodeCurFrame = np.array(encodeCurFrame)
        matches = face_recognition.compare_faces(encoded_imgs, encodeCurFrame)
        
        if not any(matches):
            cv2.imwrite(f'{imgs_path}/{newusername}_{newuserid}.jpg', frame)
            print('Image Saved')
            break
        else:
            print('Match found please try again ...')
            continue

    cv2.destroyAllWindows()
    print('Training Model')
    encode(imgs_path)
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=total_entries(),datetoday2=datetoday_formal()) 

if __name__ == '__main__':

    url = input('Enter the url of the camera: ')
    url = url+'/shot.jpg'
    app.run(host='0.0.0.0')