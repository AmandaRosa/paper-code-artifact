from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime, CheckConstraint
from datetime import datetime

DATABASE_URL = "postgresql://usuario:senha@localhost/main_server"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class EmbeddedResult(Base):

    def __init__():
        DATABASE_URL = "postgresql://usuario:senha@localhost/main_server"
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base = declarative_base()
        Base.metadata.create_all(bind=engine)
        session = SessionLocal()

    def populate_table():
        __tablename__ = "embedded_results"

        id = Column(Integer, primary_key=True, index=True)
        timestamp = Column(DateTime, default=datetime.utcnow)
        id_embarcado = Column(String, nullable=False)
        resultado_classificacao = Column(Integer, CheckConstraint("resultado_classificacao IN (0,1,2,3)"))
        uso_ram = Column(Float)
        uso_cpu = Column(Float)
        tempo_classificacao = Column(Float)
        acuracia = Column(Float)
        recall = Column(Float)


def insert_result(session, response_data):
    result_data = response_data[0]  
    new_result = object.populate_table(
        timestamp=datetime.utcnow(),
        id_embarcado=result_data["id_embarcado"],
        resultado_classificacao=result_data["result"],
        uso_ram=result_data["ram"],
        uso_cpu=result_data["cpu"],
        tempo_classificacao=result_data["tempo_classificacao"],
        acuracia=result_data["acuracia"],
        recall=result_data["recall"]
    )
    session.add(new_result)
    session.commit()
    session.refresh(new_result)
    return new_result


