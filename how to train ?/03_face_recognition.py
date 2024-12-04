import cv2
import numpy as np
import os 
import urllib.request
import paho.mqtt.client as paho
from paho import mqtt
import time
import requests

### khoi tao MQTT 


### phan MQTT

# setting callbacks for different events to see if it works, print the message etc.
def on_connect(client, userdata, flags, rc, properties=None):
    print("CONNACK received with code %s." % rc)

# with this callback you can see if your publish was successful
def on_publish(client, userdata, mid, properties=None):
    print("mid: " + str(mid))

# print which topic was subscribed to
def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

# print message, useful for checking if it was successful
def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))

client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)
client.on_connect = on_connect

# enable TLS for secure connection
client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
# set username and password
client.username_pw_set("banmuoi", "tuan2772") # vua publish vua subcribe
# connect to HiveMQ Cloud on port 8883 (default for MQTT)
client.connect("3136a68249ee49ab841675904a835fd2.s2.eu.hivemq.cloud", 8883)

# setting callbacks, use separate functions like above for better visibility
#client.on_subscribe = on_subscribe
client.on_message = on_message
client.on_publish = on_publish

###################### KET THUC MQTT


# ESP32-CAM URL
url = 'http://192.168.137.128/cam-lo.jpg'

# Load LBPH computed model (trained faces)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load the face cascade classifier
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Tuan', 'Lan']

while True:

    response = requests.get(url) # nhan anh tu URL
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, 
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
       
        # Check if confidence is less than 100 ==> "0" is a perfect match 
        if confidence < 100:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)  
         # Publishing to MQTT inside the loop
         
        if id == 'Tuan':
            time.sleep(1)
            client.publish("MungBanVeNha", payload=str(id), qos=1)
        else:
            time.sleep(1)
            client.publish("NguoiLaSamNham", payload=str('CanhBao'), qos=1)
    
    cv2.imshow('camera', img) 
  
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cv2.destroyAllWindows()
