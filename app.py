#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:/Users/SiD/Downloads/MINIPROJECT FINAL1/MINIPROJECT FINAL/keras_model.h5", "C:/Users/SiD/Downloads/MINIPROJECT FINAL1/MINIPROJECT FINAL/labels.txt")

offset = 20
imgSize = 300

labels = ["Dislike", "Goodjob", "Goodluck", "Hangout", "Hello", "Iloveyou", "Loser", "No", "Yes"]

class HandGestureRecognition(VideoProcessorBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, _ = classifier.getPrediction(imgWhite, draw=False)
                index = np.argmax(prediction)  # Get the index of the highest confidence prediction
                accuracy = round(prediction[index] * 100, 2)
                text = f"{labels[index]} - {accuracy}%"

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, _ = classifier.getPrediction(imgWhite, draw=False)
                index = np.argmax(prediction)  # Get the index of the highest confidence prediction
                accuracy = round(prediction[index] * 100, 2)
                text = f"{labels[index]} - {accuracy}%"

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, text, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)
        return imgOutput

def main():
    # Hand Gesture Recognition Application
    st.title("Real Time Hand Gesture Recognition Application")
    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam and detect your hand gestures")
    
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=HandGestureRecognition)

    st.sidebar.header("About")
    st.sidebar.markdown(
        "Developed by Sidharth J"
    )
    
    st.sidebar.header("Model Information")
    st.sidebar.text("Model: Custom Keras Model")
    st.sidebar.text("Labels: " + ", ".join(labels))

if __name__ == "__main__":
    main()


# In[ ]:




