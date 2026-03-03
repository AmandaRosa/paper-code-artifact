import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from filter import LowPassFilter
import numpy as np
import random

def read_data(directory_path, dict_data):

    array_data = list()

    arrays = [[] for _ in range(0, 9)]
    i=0
    for root, dirs, files in os.walk(directory_path):
        print(files)
        key = os.path.relpath(root, start="../data")
        for file in sorted(files):
            if i==0:
                i+=1
                if file.endswith(".txt") or file.endswith(".csv"):
                    print(file)
                    file_path = os.path.join(root, file)

                    with open(file_path, "r") as f:
                        lines = f.readlines()
                        filtered_data = [item.strip() for item in lines if item.strip()]
                        for item in filtered_data:
                            array_info = item.split(";")
                            for i in range(0, 9):
                                arrays[i].append(float(array_info[i]))
                        # array_data_normal1 = (np.array(arrays))
                        # print(array_data_normal1.shape)

    array_data = np.array(arrays)
    dict_data[key] = array_data.T
    return dict_data

def add_noise(signal, percentage=0.025):
    """
    Add Gaussian noise to the input signal based on a percentage of the signal's amplitude.

    Parameters:
        signal (numpy.ndarray): Input signal.
        percentage (float): Percentage of the signal's amplitude to use as the noise level.

    Returns:
        numpy.ndarray: Noisy signal.
    """
    if percentage < 0 or percentage > 1:
        raise ValueError("Percentage must be between 0 and 1")

    noise = np.random.normal(scale=1, size=len(signal))
    noise_level = percentage * np.sqrt(np.mean((signal - np.mean(signal)) ** 2))
    noisy_signal = signal + noise_level * noise
    return noisy_signal

class Samples:

    def __init__(self):

        self.dict_data = {}
        self.rodada = 0
        self.turn = -1

        self.dict_data_final = read_data("../data/Normal_1", self.dict_data)
        self.dict_data_final = read_data("../data/Normal_2", self.dict_data)
        self.dict_data_final = read_data("../data/Normal_3", self.dict_data)
        self.dict_data_final = read_data("../data/Normal_4", self.dict_data)
        self.dict_data_final = read_data("../data/Unbalance_6g/", self.dict_data)
        self.dict_data_final = read_data("../data/Unbalance_10g/", self.dict_data)
        self.dict_data_final = read_data("../data/Unbalance_15g/", self.dict_data)
        self.dict_data_final = read_data("../data/Unbalance_20g/", self.dict_data)
        self.dict_data_final = read_data("../data/Unbalance_25g/", self.dict_data)
        self.dict_data_final = read_data("../data/Unbalance_30g/", self.dict_data)
        self.dict_data_final = read_data("../data/Horizontal_Misalignment_0_5mm/", self.dict_data)
        self.dict_data_final = read_data("../data/Horizontal_Misalignment_1mm/", self.dict_data)
        self.dict_data_final = read_data("../data/Horizontal_Misalignment_1_5mm/", self.dict_data)
        self.dict_data_final = read_data("../data/Horizontal_Misalignment_2mm/", self.dict_data)
        self.dict_data_final = read_data("../data/Vertical_Misalignment_0_51mm/", self.dict_data)
        self.dict_data_final = read_data("../data/Vertical_Misalignment_1_27mm/", self.dict_data)
        self.dict_data_final = read_data("../data/Vertical_Misalignment_1_4mm/", self.dict_data)
        self.dict_data_final = read_data("../data/Vertical_Misalignment_1_78mm/", self.dict_data)
        self.dict_data_final = read_data("../data/Vertical_Misalignment_1_91mm/", self.dict_data)

        self.channel = 'ch8'

        self.channels = ['ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
        self.normal1_signal = {channel: [] for channel in self.channels}
        self.normal2_signal = {channel: [] for channel in self.channels}
        self.normal3_signal = {channel: [] for channel in self.channels}
        self.normal4_signal = {channel: [] for channel in self.channels}
        self.unbalanced6g_signal = {channel: [] for channel in self.channels}
        self.unbalanced10g_signal = {channel: [] for channel in self.channels}
        self.unbalanced15g_signal = {channel: [] for channel in self.channels}
        self.unbalanced20g_signal = {channel: [] for channel in self.channels}
        self.unbalanced25g_signal = {channel: [] for channel in self.channels}
        self.unbalanced30g_signal = {channel: [] for channel in self.channels}
        self.horizontalmisalign_0_5mm_signal = {channel: [] for channel in self.channels}
        self.horizontalmisalign_1mm_signal = {channel: [] for channel in self.channels}
        self.horizontalmisalign_1_5mm_signal = {channel: [] for channel in self.channels}
        self.horizontalmisalign_2mm_signal = {channel: [] for channel in self.channels}
        self.verticalmisalign_0_51mm_signal = {channel: [] for channel in self.channels}
        self.verticalmisalign_1_27mm_signal = {channel: [] for channel in self.channels}
        self.verticalmisalign_1_4mm_signal = {channel: [] for channel in self.channels}
        self.verticalmisalign_1_78mm_signal = {channel: [] for channel in self.channels}
        self.verticalmisalign_1_91mm_signal = {channel: [] for channel in self.channels}
        

        for i in range(2,9,1):

            for array in self.dict_data_final["Normal_1"]:
                self.normal1_signal[f'ch{i}'].append(array[i-1])  # Assuming column 1 is time_vector

            for array in self.dict_data_final["Normal_2"]:
                self.normal2_signal[f'ch{i}'].append(array[i-1])  # Assuming column 1 is time_vector

            for array in self.dict_data_final["Normal_3"]:
                self.normal3_signal[f'ch{i}'].append(array[i-1])  # Assuming column 1 is time_vector

            for array in self.dict_data_final["Normal_4"]:
                self.normal4_signal[f'ch{i}'].append(array[i-1])  # Assuming column 1 is time_vector

            for array in self.dict_data_final["Unbalance_6g"]:
                self.unbalanced6g_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Unbalance_10g"]:
                self.unbalanced10g_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Unbalance_15g"]:
                self.unbalanced15g_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector
            
            for array in self.dict_data_final["Unbalance_20g"]:
                self.unbalanced20g_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Unbalance_25g"]:
                self.unbalanced25g_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Unbalance_30g"]:
                self.unbalanced30g_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Horizontal_Misalignment_0_5mm"]:
                self.horizontalmisalign_0_5mm_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Horizontal_Misalignment_1mm"]:
                self.horizontalmisalign_1mm_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Horizontal_Misalignment_1_5mm"]:
                self.horizontalmisalign_1_5mm_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Horizontal_Misalignment_2mm"]:
                self.horizontalmisalign_2mm_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Vertical_Misalignment_0_51mm"]:
                self.verticalmisalign_0_51mm_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Vertical_Misalignment_1_27mm"]:
                self.verticalmisalign_1_27mm_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Vertical_Misalignment_1_4mm"]:
                self.verticalmisalign_1_4mm_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Vertical_Misalignment_1_78mm"]:
                self.verticalmisalign_1_78mm_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector

            for array in self.dict_data_final["Vertical_Misalignment_1_91mm"]:
                self.verticalmisalign_1_91mm_signal[f'ch{i}'].append(array[i-1]) # Assuming column 1 is time_vector
        
        self.noise_values = np.linspace(0.001, 1, 11) ##aqui altera o nível de ruído a ser inserido no sistema
        # self.noise_values = np.linspace(0.4006, 1, 7) ##aqui altera o nível de ruído a ser inserido no sistema

        self.normal1_signal_list = {channel: [] for channel in self.channels}
        self.normal2_signal_list = {channel: [] for channel in self.channels}
        self.normal3_signal_list = {channel: [] for channel in self.channels}
        self.normal4_signal_list = {channel: [] for channel in self.channels}
        self.unbalanced6g_signal_list = {channel: [] for channel in self.channels}
        self.unbalanced10g_signal_list = {channel: [] for channel in self.channels}
        self.unbalanced15g_signal_list = {channel: [] for channel in self.channels}
        self.unbalanced20g_signal_list = {channel: [] for channel in self.channels}
        self.unbalanced25g_signal_list = {channel: [] for channel in self.channels}
        self.unbalanced30g_signal_list = {channel: [] for channel in self.channels}
        self.horizontalmisalign_0_5mm_signal_list = {channel: [] for channel in self.channels}
        self.horizontalmisalign_1mm_signal_list = {channel: [] for channel in self.channels}
        self.horizontalmisalign_1_5mm_signal_list = {channel: [] for channel in self.channels}
        self.horizontalmisalign_2mm_signal_list = {channel: [] for channel in self.channels}
        self.verticalmisalign_0_51mm_signal_list = {channel: [] for channel in self.channels}
        self.verticalmisalign_1_27mm_signal_list = {channel: [] for channel in self.channels}
        self.verticalmisalign_1_4mm_signal_list = {channel: [] for channel in self.channels}
        self.verticalmisalign_1_78mm_signal_list = {channel: [] for channel in self.channels}
        self.verticalmisalign_1_91mm_signal_list = {channel: [] for channel in self.channels}


        for i in range(2,9,1):
            self.normal1_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.normal2_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.normal3_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.normal4_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.unbalanced6g_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.unbalanced10g_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.unbalanced15g_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.unbalanced20g_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.unbalanced25g_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.unbalanced30g_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.horizontalmisalign_0_5mm_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.horizontalmisalign_1mm_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.horizontalmisalign_1_5mm_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.horizontalmisalign_2mm_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.verticalmisalign_0_51mm_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.verticalmisalign_1_27mm_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.verticalmisalign_1_4mm_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.verticalmisalign_1_78mm_signal_list[f'ch{i}'] = [None] * len(self.noise_values)
            self.verticalmisalign_1_91mm_signal_list[f'ch{i}'] = [None] * len(self.noise_values)


        sample = 0
        for noise_level in self.noise_values:
            for i in range(2,9,1):
                # self.normal1_signal_list[f'ch{i}'][sample] = LowPassFilter(add_noise(self.normal1_signal[f'ch{i}'], noise_level))
                # self.unbalanced30g_signal_list[f'ch{i}'][sample] = LowPassFilter(add_noise(self.unbalanced30g_signal[f'ch{i}'], noise_level))
                # self.horizontalmisalign_2mm_signal_list[f'ch{i}'][sample] = LowPassFilter(add_noise(self.horizontalmisalign_2mm_signal[f'ch{i}'], noise_level))
                # self.verticalmisalign_1_27mm_signal_list[f'ch{i}'][sample] = LowPassFilter(add_noise(self.verticalmisalign_1_27mm_signal[f'ch{i}'], noise_level))
                self.normal1_signal_list[f'ch{i}'][sample] = add_noise(self.normal1_signal[f'ch{i}'], noise_level)
                self.normal2_signal_list[f'ch{i}'][sample] = add_noise(self.normal2_signal[f'ch{i}'], noise_level)
                self.normal3_signal_list[f'ch{i}'][sample] = add_noise(self.normal3_signal[f'ch{i}'], noise_level)
                self.normal4_signal_list[f'ch{i}'][sample] = add_noise(self.normal4_signal[f'ch{i}'], noise_level)
                self.unbalanced6g_signal_list[f'ch{i}'][sample] = add_noise(self.unbalanced6g_signal[f'ch{i}'], noise_level)
                self.unbalanced10g_signal_list[f'ch{i}'][sample] = add_noise(self.unbalanced10g_signal[f'ch{i}'], noise_level)
                self.unbalanced15g_signal_list[f'ch{i}'][sample] = add_noise(self.unbalanced15g_signal[f'ch{i}'], noise_level)
                self.unbalanced20g_signal_list[f'ch{i}'][sample] = add_noise(self.unbalanced20g_signal[f'ch{i}'], noise_level)
                self.unbalanced25g_signal_list[f'ch{i}'][sample] = add_noise(self.unbalanced25g_signal[f'ch{i}'], noise_level)
                self.unbalanced30g_signal_list[f'ch{i}'][sample] = add_noise(self.unbalanced30g_signal[f'ch{i}'], noise_level)
                self.horizontalmisalign_0_5mm_signal_list[f'ch{i}'][sample] = add_noise(self.horizontalmisalign_0_5mm_signal[f'ch{i}'], noise_level)
                self.horizontalmisalign_1mm_signal_list[f'ch{i}'][sample] = add_noise(self.horizontalmisalign_1mm_signal[f'ch{i}'], noise_level)
                self.horizontalmisalign_1_5mm_signal_list[f'ch{i}'][sample] = add_noise(self.horizontalmisalign_1_5mm_signal[f'ch{i}'], noise_level)
                self.horizontalmisalign_2mm_signal_list[f'ch{i}'][sample] = add_noise(self.horizontalmisalign_2mm_signal[f'ch{i}'], noise_level)
                self.verticalmisalign_0_51mm_signal_list[f'ch{i}'][sample] = add_noise(self.verticalmisalign_0_51mm_signal[f'ch{i}'], noise_level)
                self.verticalmisalign_1_27mm_signal_list[f'ch{i}'][sample] = add_noise(self.verticalmisalign_1_27mm_signal[f'ch{i}'], noise_level)
                self.verticalmisalign_1_4mm_signal_list[f'ch{i}'][sample] = add_noise(self.verticalmisalign_1_4mm_signal[f'ch{i}'], noise_level)
                self.verticalmisalign_1_78mm_signal_list[f'ch{i}'][sample] = add_noise(self.verticalmisalign_1_78mm_signal[f'ch{i}'], noise_level)
                self.verticalmisalign_1_91mm_signal_list[f'ch{i}'][sample] = add_noise(self.verticalmisalign_1_91mm_signal[f'ch{i}'], noise_level)           
            sample+=1

    def escolher_chave(self, dict_disparo):
            # Definindo as chaves
            chaves = list(dict_disparo.keys())
            
            # Definindo as probabilidades, com 80% para 'Normal'
            probabilidades = [0.3 if "Normal" in chave else 0.7 / (len(chaves) - 1) for chave in chaves]
            
            # Escolhendo uma chave aleatoriamente com base nas probabilidades
            chave_escolhida = random.choices(chaves, weights=probabilidades, k=1)[0]
            
            # Retornando a chave e o valor correspondente
            return chave_escolhida, dict_disparo[chave_escolhida]
    
    
    def disparar(self):


        if self.rodada%100 == 0:
            self.turn+=1

        print(self.turn)

        dict_disparo = {
            'Normal_1': self.normal2_signal_list[self.channel][self.turn],
            'Normal_2': self.normal2_signal_list[self.channel][self.turn],
            'Normal_3': self.normal3_signal_list[self.channel][self.turn],
            'Normal_4': self.normal4_signal_list[self.channel][self.turn],
            'Unbalanced_6g': self.unbalanced6g_signal_list[self.channel][self.turn],
            'Unbalanced_10g': self.unbalanced10g_signal_list[self.channel][self.turn],
            'Unbalanced_15g': self.unbalanced15g_signal_list[self.channel][self.turn],
            'Unbalanced_20g': self.unbalanced20g_signal_list[self.channel][self.turn],
            'Unbalanced_25g': self.unbalanced25g_signal_list[self.channel][self.turn],
            'Unbalanced_30g': self.unbalanced30g_signal_list[self.channel][self.turn],
            'Horizontal_Misalignment_0_5mm': self.horizontalmisalign_0_5mm_signal_list[self.channel][self.turn],
            'Horizontal_Misalignment_1mm': self.horizontalmisalign_1mm_signal_list[self.channel][self.turn],
            'Horizontal_Misalignment_1_5mm': self.horizontalmisalign_1_5mm_signal_list[self.channel][self.turn],
            'Horizontal_Misalignment_2mm': self.horizontalmisalign_2mm_signal_list[self.channel][self.turn],
            'Vertical_Misalignment_0_51mm': self.verticalmisalign_0_51mm_signal_list[self.channel][self.turn],
            'Vertical_Misalignment_1_27mm': self.verticalmisalign_1_27mm_signal_list[self.channel][self.turn],
            'Vertical_Misalignment_1_4mm': self.verticalmisalign_1_4mm_signal_list[self.channel][self.turn],
            'Vertical_Misalignment_1_78mm': self.verticalmisalign_1_78mm_signal_list[self.channel][self.turn],
            'Vertical_Misalignment_1_91mm': self.verticalmisalign_1_91mm_signal_list[self.channel][self.turn],
        }

        chave, valor = self.escolher_chave(dict_disparo)
        self.rodada+=1
        return chave,valor