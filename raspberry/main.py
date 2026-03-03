from fastapi import FastAPI
from subscriber import Subscriber
from framework import FrameworkSystem
import sqlite3
from fastapi.responses import JSONResponse
import threading
import time
from contextlib import asynccontextmanager


# Instancia os objetos fora das rotas para reuso
subscriber = Subscriber()
session = FrameworkSystem()

# def call_functions_periodically():
#     while True:
#         try:
#             get_mensagem()
#         except Exception as e:
#             print("Erro ao chamar funções:", e)

#         time.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    subscriber_thread = threading.Thread(target=subscriber.run_forever, daemon=True)
    subscriber_thread.start()

    print("Thread de background iniciada.")

    yield  # Aqui o app está rodando

    # Shutdown (se quiser algo ao encerrar, coloque aqui)
    print("App encerrando...")

# Inicializa o app FastAPI
app = FastAPI(lifespan=lifespan)

# Rota GET raiz
@app.get("/")
def home():
    return {"mensagem": "Olá, Amanda!"}

# Rota GET amanda
@app.get("/amanda")
def amanda():
    print("Chamando Amanda internamente")
    return {"mensagem": "Chamando Amanda!"}

# # Rota GET ler_fila
# @app.get("/ler_fila")
# def get_mensagem():
#     subscriber.calling_subscriber()
#     return {"status": "Fila lida"}

@app.get("/get_results")
def get_results():
    try:
        conn = sqlite3.connect('raspberry_database.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM embedded_results ORDER BY timestamp ASC')
        rows = cursor.fetchall()

        resultados = [dict(row) for row in rows]
        return JSONResponse(content=resultados)

    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)

    finally:
        conn.close()
