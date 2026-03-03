from publisher import *
from samples import *
# from database import Base
# from database import SessionLocal, engine, insert_result, EmbeddedResult
import requests

publisher = Publisher()
amostras = Samples()
# embedded_result = EmbeddedResult()

while True:
    chave, valor = amostras.disparar()
    topic = "test/topic1"
    message = {f'{chave}': valor}
    publisher.publish(topic, message)
    # response = requests.get("http://localhost:8000/get_results") ##localhost será substituido pelo IP da raspberry!!!
    # print(response.json())
    # response = requests.get("http://192.168.100.23:8000/get_results")
    # if response.status_code == 200:
    #     data = response.json()  
    #     insert_result(embedded_result.session, data)
    # else:
    #     print(f"Erro na requisição: {response.status_code}")