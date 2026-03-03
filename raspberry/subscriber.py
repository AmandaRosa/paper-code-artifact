import paho.mqtt.client as mqtt
import json
from framework import FrameworkSystem

session = FrameworkSystem()

class Subscriber:
    def __init__(self):
        self.topic = 'test/topic1'
        # self.broker_address = "10.1.35.234"
        self.broker_address = "localhost"
        self.client = mqtt.Client(client_id="Subscriber")

        # Callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT] Conectado ao broker {self.broker_address}")
            client.subscribe(self.topic)
        else:
            print(f"[MQTT] Falha na conexão. Código de retorno: {rc}")

    def on_disconnect(self, client, userdata, rc):
        print(f"[MQTT] Desconectado. Código: {rc}. Tentando reconectar...")
        try:
            client.reconnect()
        except Exception as e:
            print(f"[MQTT] Falha ao reconectar: {e}")

    def on_message(self, client, userdata, message):
        try:
            mqtt_msg = message.payload.decode()
            # print(f"[MQTT] Mensagem recebida: {mqtt_msg}")
            data_dict = json.loads(mqtt_msg)
            session.run(data_dict)
        except Exception as e:
            print(f"[MQTT] Erro ao processar mensagem: {e}")

    def run_forever(self):
        try:
            self.client.connect(self.broker_address)
            self.client.loop_start()  # ✅ NÃO bloqueia
        except Exception as e:
            print(f"[MQTT] Erro ao conectar: {e}")
