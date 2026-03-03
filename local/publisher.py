import paho.mqtt.client as mqtt
import time
import json
import numpy as np


class Publisher:

    def __init__(self):

        # self.broker_address = "10.1.35.234"  ## ao inves de usar localhost vou usar ip do notebook ifconfig
        self.broker_address = "localhost"  ## ao inves de usar localhost vou usar ip do notebook ifconfig
        self.broker_port = 1883
        self.topics = ["test/topic1", "test/topic2", "test/topic3"]
        self.client = mqtt.Client(client_id="Publisher")
        self.client.connect(self.broker_address, 1883)


    def publish(self, topic, message):
        key = (list(message.keys())[0])
        print(key)
        data = message[key].tolist()
        payload = {key: data}
        mqtt_msg = json.dumps(payload)
        self.client.publish(topic, mqtt_msg)
        # print(f"Published '{mqtt_msg}' to topic '{topic}'")
        time.sleep(70) 

