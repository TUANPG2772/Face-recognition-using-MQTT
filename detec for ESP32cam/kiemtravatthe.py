import cv2
import urllib.request
import numpy as np
import paho.mqtt.client as paho
from paho import mqtt
import time



url = 'http://192.168.137.92/cam-lo.jpg'
winName = 'ESP32 CAMERA'
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

counter = 0
temp = 0 
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

while True:
    imgResponse = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    if len(classNames) > 0 and len(classIds) != 0:
        time.sleep(2)
        counter = len(classIds)
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if 0 <= classId-1 < len(classNames):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                cv2.putText(img, classNames[classId-1], (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                if classNames[classId-1] == 'personal' :
                        temp = temp + 1
                #else: temp = 0 # check
    else:
        counter = 0
        temp = 0 


    print("temp :", temp)
    cv2.imshow(winName, img)
    
    
    #client.publish("testTopic", "Hello, ESP8266")
    client.publish("SoVatThe", payload=str(counter), qos=1)


 # Fixed publish syntax here

    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
cv2.destroyAllWindows()
