import os
import glob
import numpy as np
import pandas as pd
import re
import statistics
import pickle
import sqlite3
import psutil
from utils.methods import PcaApplication, MeanApplication
from datetime import datetime
from utils.filter import LowPassFilter
from utils.timedomain import TimeDomain
from utils.frequencydomain import FrequencyDomain
from utils.timefrequencydomain_predict import TimeFrequencyDomain
from scipy.signal import hilbert
from subscriber import *

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class FrameworkSystem(metaclass=SingletonMeta):
    def __init__(self):
        self._frameworks = {}
        self.create_db()
        self.embarcado_id = '1234'
        self.label_dict = {0: 'Normal', 1: 'Unbalanced', 2:'Horizontal_Misalignment', 3:'Vertical_Misalignment'}
        self.noise_values = np.linspace(0.001, 1, 11)
        # self.noise_values = np.linspace(0.4006, 1, 7)
        self.acertos = 0
        self.rodada = 0
        self.execucao_noise = 0
        self.turn = 0
        self.noise_percentage = 0
        self.excel_results = list()

    def create_db(self):

        conn = sqlite3.connect('raspberry_database.db')
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS embedded_results (
            id INTEGER PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            id_embarcado TEXT NOT NULL,
            resultado_classificacao INTEGER CHECK(resultado_classificacao IN (0,1,2,3)),
            uso_ram REAL,
            uso_cpu REAL,
            tempo_classificacao REAL,
            acuracia REAL,
            recall REAL
        )
        ''')

        conn.commit()
        conn.close()

    def clear_pycache(self):
        pycache_files = glob.glob('__pycache__/*.pyc')
        for file in pycache_files:
            os.remove(file)
    
    def pre_process(self, label, signal):

        signals_filtered = LowPassFilter(signal)
        data = {f'{label}':signals_filtered}
        return data

    def reference_process(self, data, sample_size):        
        normal_signal = data['Normal'][sample_size:2*sample_size]
        signal_reference = LowPassFilter(normal_signal)
        analytic_signal = hilbert(signal_reference)
        envelope = np.abs(analytic_signal)
        rolling_mean = pd.Series(envelope).rolling(window=2500).mean()
        rolling_std = list(pd.Series(envelope).rolling(window=2500).std())[-1]
        adaptive_threshold = rolling_mean + 3 * rolling_std
        original_peak = np.max(normal_signal)  
        adaptive_mean = np.mean(adaptive_threshold[:2500])  
        correction_factor = original_peak / adaptive_mean  
        fixed_threshold = adaptive_mean * correction_factor  
        std_positive = fixed_threshold + 2* rolling_std
        std_negative = fixed_threshold - 2* rolling_std

        return fixed_threshold, std_positive, std_negative

    def classification(self, sinal, channel):
        signal_processed = {}
        predictions=[]

        features_function = {
                        'Root_Mean_Square': self.time_domain_instance.root_mean_square, 
                        'Std': self.time_domain_instance.std, 
                        'Average_Mean_from_Envelope': self.time_domain_instance.average_mean_from_envelope, 
                        'Densidade_Espectral_Potencia': self.frequency_domain_instance.densidade_espectral_potencia,
                        'Spectrogram_HOG': self.timefrequency_domain_instance.spectrogram_hog,
                        'Spectrogram_LBP': self.timefrequency_domain_instance.spectrogram_lbp,
                        # 'Spectrogram_HISTOGRAM': timefrequency_domain_instance.spectrogram_histogram,
                        # 'Spectrogram_HIST_MEAN_SKEWNESS': timefrequency_domain_instance.spectrogram_hist_mean_skewness,
                        'Short_Time_Fourier_Transform_HOG': self.timefrequency_domain_instance.short_time_fourier_transform_hog,
                        'Short_Time_Fourier_Transform_LBP': self.timefrequency_domain_instance.short_time_fourier_transform_lbp,
                        # 'Short_Time_Fourier_Transform_HISTOGRAM': timefrequency_domain_instance.short_time_fourier_transform_histogram, 
                        # 'Short_Time_Fourier_Transform_HIST_MEAN_SKEWNESS': timefrequency_domain_instance.short_time_fourier_transform_hist_mean_skewness        
        }

        tf_methods = ['Spectrogram_HOG', 'Spectrogram_LBP', 'Spectrogram_HISTOGRAM', 'Spectrogram_HIST_MEAN_SKEWNESS', 'Wavelet_Transform_LBP',
                    'Wavelet_Transform_HOG', 'Wavelet_Transform_HIST_MEAN_SKEWNESS',
                    'Wavelet_Transform_HISTOGRAM', 'Short_Time_Fourier_Transform_HOG', 
                    'Short_Time_Fourier_Transform_HIST_MEAN_SKEWNESS', 'Short_Time_Fourier_Transform_LBP', 
                    'Short_Time_Fourier_Transform_HISTOGRAM']

        for key, func in features_function.items(): 
            if key in tf_methods:
                # aux = func(sinal) 
                aux = func(sinal, 5) 
            else:
                aux = func(sinal) 
            signal_processed[key] = np.array(aux)
            if isinstance(signal_processed[key], np.ndarray):
                if signal_processed[key].ndim == 2:
                    if signal_processed[key].shape[1] > 1:
                        if key in tf_methods:
                            signal_processed[key] = MeanApplication(signal_processed[key])   
                        else:
                            signal_processed[key] = PcaApplication(signal_processed[key], 1)

        base_paths = ["./models/T-F-models/", "./models/TF-models/"]
        available_files = []

        for path in base_paths:
            if os.path.exists(path):
                available_files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pkl")])

        available_files_channel = [i for i in available_files if channel in i]

        models = {}
        aux = {}
        resultados = []

        for i, model_path in enumerate(available_files_channel):
            with open(f"{model_path}", 'rb') as file:
                models[f'{i}'] = pickle.load(file)
                match = re.search(r'_(RandomForest|MLP|SVM)_(.+)\.pkl$', model_path)
                if match:
                    method_name = match.group(2)

                    if method_name in ['Root_Mean_Square', 'Std', 'Average_Mean_from_Envelope', 'Densidade_Espectral_Potencia','Spectriogram_HOG', 'Spectrogram_LBP']:
                        aux[f'{i}'] = models[f'{i}'].predict(signal_processed[f'{method_name}'].reshape(-1,1))

                    elif method_name in ['Wavelet_Transform_LBP','Wavelet_Transform_HOG', 'Wavelet_Transform_HIST_MEAN_SKEWNESS',
                                    'Wavelet_Transform_HISTOGRAM', 'Short_Time_Fourier_Transform_HOG', 
                                    'Short_Time_Fourier_Transform_HIST_MEAN_SKEWNESS', 'Short_Time_Fourier_Transform_LBP', 
                                    'Short_Time_Fourier_Transform_HISTOGRAM']:

                        aux[f'{i}'] = models[f'{i}'].predict(signal_processed[f'{method_name}'][0].reshape(-1,1))
                    else:
                        aux[f'{i}'] = models[f'{i}'].predict(signal_processed[f'{method_name}'])

        for i in range(0, len(aux)):
            if len(aux[f'{i}'])== 1:
                resultados.append(aux[f'{i}'])
            else:
                resultados.extend(aux[f'{i}'])

        predictions = [
        float(x[0]) if isinstance(x, (np.ndarray, list)) else float(x)  
        for x in resultados
        ]

        return predictions
        
    def run(self, mensagem):
        self.result = []
        starttime = datetime.now()
        self.clear_pycache()

        true_label = list(mensagem.keys())[0]
        signal = mensagem[true_label]

        data = self.pre_process(true_label, signal)

        print('dado processado: ', data.keys())

        if list(data.keys())[0] == 'Normal':
            fixed_threshold, std_positive, std_negative = self.reference_process(data, 2500)
            print('threshold: ', fixed_threshold)
            print('std_positive: ', std_positive)
            print('std_negative: ', std_negative)

        min_sample_length = 25000
        self.time_domain_instance = TimeDomain(sample_length=min_sample_length, subsample=min_sample_length )
        self.frequency_domain_instance = FrequencyDomain(sample_length=min_sample_length , pontos = min_sample_length )
        self.timefrequency_domain_instance = TimeFrequencyDomain(sample_length=min_sample_length , subsample=min_sample_length)
        classification_detection = list()

        classification_detection.extend([0] *len(signal))
        signal_length = len(signal) #250.000
        chunk_size = min_sample_length

        for start in range(0, signal_length, chunk_size):
            end = start + chunk_size
            signal_sample = np.array(signal[start:end])
            predicao_sistema = self.classification(signal_sample, 'ch8') 

            moda_predicao_final = statistics.mode(predicao_sistema)

            self.result.append(moda_predicao_final)
            # classification_detection[start:end]= [moda_predicao_final] * (end - start)

            ## salvar classificacao 25k por 25k? ou salvar a classificação geral do sinal todo de 125k?
        final_result = statistics.mode(self.result)

        if self.rodada%100 == 0:
            self.acertos = 0

        if self.label_dict[final_result] in true_label:
            self.acertos +=1

        if self.rodada%100 == 0:
            self.noise_percentage = self.noise_values[self.turn]
            self.turn+=1
            self.execucao_noise = 0
        self.rodada +=1
        self.execucao_noise +=1

        acuracia = self.acertos/self.execucao_noise

        self.write_result_excel(self.noise_percentage, self.label_dict[final_result], true_label, self.acertos, acuracia)

        print(self.noise_percentage)
        print("Predito:", self.label_dict[final_result])
        print("Real:", true_label)
        print('ACERTOS:', self.acertos)
        print("Acuracia:", acuracia)
        endtime = datetime.now()
        process_time_in_seconds = (endtime - starttime).total_seconds()

        cpu_percent = psutil.cpu_percent(interval=1) 
        mem = psutil.virtual_memory()
        memory_used = mem.used / (1024 ** 2)
        total_memory = mem.total / (1024 ** 2)
        ram = psutil.virtual_memory().percent

        self.insert_db(final_result, ram, cpu_percent, process_time_in_seconds, acuracia)


    def insert_db(self,result, ram, cpu, tempo_classificacao, acuracia):

        timestamp = datetime.now().isoformat(sep=' ')
        recall = 0.88 #### corrigir aqui

        # Connect to your database
        conn = sqlite3.connect('raspberry_database.db')
        cursor = conn.cursor()

        # Insert data
        cursor.execute('''
            INSERT INTO embedded_results (
                timestamp, id_embarcado, resultado_classificacao,
                uso_ram, uso_cpu, tempo_classificacao, acuracia, recall
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, self.embarcado_id, result,
            ram, cpu, tempo_classificacao, acuracia, recall
        ))

        conn.commit()
        conn.close()

    def write_result_excel(self, noise_percentage, predicted, true_label, acertos, acuracia, file_path='resultados_noise_percentage.xlsx'):
        row = {
            'Noise_Percentage': noise_percentage,
            'Predito': predicted,
            'Real': true_label,
            'Acertos': acertos,
            'Acuracia': acuracia
        }

        # If file exists, append without headers
        if os.path.exists(file_path):
            df = pd.DataFrame([row])
            with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        else:
            df = pd.DataFrame([row])
            df.to_excel(file_path, index=False)






        