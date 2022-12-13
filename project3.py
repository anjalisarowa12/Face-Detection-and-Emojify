from ast import Return
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import adam_v2
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import webbrowser
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model = keras.models.load_model('/Users/anjalisarowa/Desktop/modelsave/model2')
cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: "    Angry   ", 1: "Disgust", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ",5: "    Sad    ", 6: "Surprised"}
emoji_dist={0:"/Users/anjalisarowa/Desktop/modelsave/emojis/angry.jpeg",1:"/Users/anjalisarowa/Desktop/modelsave/emojis/disgust.jpeg",
            2:"/Users/anjalisarowa/Desktop/modelsave/emojis/fearful.jpeg",3:"/Users/anjalisarowa/Desktop/modelsave/emojis/happy.jpeg",
            4:"/Users/anjalisarowa/Desktop/modelsave/emojis/neutral.jpeg" ,5:"/Users/anjalisarowa/Desktop/modelsave/emojis/sad.jpeg",
            6:"/Users/anjalisarowa/Desktop/modelsave/emojis/surprised.jpeg"}

          
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
cap1 = cv2.VideoCapture(0)
show_text=[0]
def show_vid():
    if not cap1.isOpened():
        print("cant open the camera1")
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1,(600,500))
    bounding_box = cv2.CascadeClassifier('/Users/anjalisarowa/opt/anaconda3/pkgs/libopencv-4.5.1-py39h2677ad0_0/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex
    if flag1 is None:
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_vid)

def show_vid2():
    frame2=cv2.imread(emoji_dist[show_text[0]])
    pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(frame2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))

    lmain2.configure(image=imgtk2)
    lmain2.after(10, show_vid2)
    

if __name__ == '__main__':
    root=tk.Tk()
    heading2=Label(root,text="Photo to Emoji ",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')
    heading2.pack()
    heading3=Label(root,text="Music Recommendation: ",pady=20, font=('arial',30,'bold'),bg='black',fg='#CDCDCD')
    heading3.place(x=100, y=100)
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2 = tk.Label(master=root,bd=10)
    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50,y=250)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900,y=350)

    def callback(url):
        webbrowser.open_new_tab(url)

    link = Label(root, text=" Happy ",font=('Helveticabold', 15), bg="blue",fg="white", pady=10, cursor="hand2")
    link.place(x=100, y=200)
    link.bind("<Button-1>", lambda e:
    callback("https://youtu.be/OYuRPH17pLs"))

    link = Label(root, text="  Sad  ",font=('Helveticabold', 15), bg="blue",fg="white", pady=10, cursor="hand2")
    link.place(x=300, y=200)
    link.bind("<Button-1>", lambda e:
    callback("https://youtube.com/playlist?list=PL3-sRm8xAzY-w9GS19pLXMyFRTuJcuUjy"))

    link = Label(root, text=" Angry ",font=('Helveticabold', 15), bg="blue",fg="white", pady=10, cursor="hand2")
    link.place(x=500, y=200)
    link.bind("<Button-1>", lambda e:
    callback("https://youtube.com/playlist?list=PL7v1FHGMOadBhCjuh_ljEEhqrQKCBsoIn"))

    link = Label(root, text="Disgust",font=('Helveticabold', 15), bg="blue", fg="white", pady=10, cursor="hand2")
    link.place(x=700,y=200)
    link.bind("<Button-1>", lambda e:
    callback("https://youtu.be/me4plodHJWI"))
       
    link = Label(root, text="Fearful",font=('Helveticabold', 15), bg="blue",fg="white", pady=10, cursor="hand2")
    link.place(x=900,y=200)
    link.bind("<Button-1>", lambda e:
    callback("https://youtu.be/1Z_deVJoIrM"))

    link= Label(root, text="Neutral",font=('Helveticabold', 15), bg="blue",fg="white", pady=10, cursor="hand2")
    link.place(x=1100,y=200)
    link.bind("<Button-1>", lambda e:
    callback("https://youtu.be/6GADpWQgsb0"))
    
            
    link = Label(root, text="Surprised",font=('Helveticabold', 15),bg="blue", fg="white", pady=10, cursor="hand2")
    link.place(x= 1300, y=200)
    link.bind("<Button-1>", lambda e:
    callback("https://youtu.be/JIlxyOXFkAs"))
    
    root.title("Photo To Emoji \n Music Recommendation")
    
    root.geometry("1400x900+100+10")
    root['bg']='black'
    exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)
    
    show_vid()
    show_vid2()
    root.mainloop()
    cap1.release()
   

   