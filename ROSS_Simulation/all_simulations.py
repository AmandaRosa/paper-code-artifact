import numpy as np
import simulate_misalignement as sm
import simulate_unbalance as smb
import simulate_rubbing as sr
import simulate_crack as sc
import pandas as pd
from multiprocessing import Process, Queue

dt = 2e-5
tf = 15
tf_desejado = 15
limit_of_debug_samples = 7
dist = 0.068


# probes 12 e 23
probes_nodes = [12,23]

# angulos das probes 0 e 90 (em radianos é 1.5708)
probes_angles = [0, 1.5708]

# velocidades de 13Hz - 780 e 60Hz - 3600 COLOCAR UMA VELOCIDADE POR VEZ
speeds = [3600]

##########################################################################################################

## possíveis magnitudes de desbalanceamento
## 5g e 30g, considerando raio de 6.8 cm
## como está em Kg*m e eu passei como g*mm temos que /10⁶
possible_unbalance_magnitudes = [((75/1000000)*dist), ((450/1000000)*dist)]

## possíveis fases de desbalanceamento [0, 90] em rad
possible_unbalance_phases = [0, 1.5708]

## possíveis massas para desbalanceamento (quando em outros defeitos)
qt_discs = 2
num_samples = 2
bottom_limit = 0.005
top_limit = 0.5
possible_mass_defects_background = [((75/1000000)*dist), ((450/1000000)*dist)]
possible_mass_misalignement = [((75/1000000)*dist), ((450/1000000)*dist)]


## possíveis massas para desbalanceamento quando rotinas de desbalanceamento
## 3 valores aleatórios entre 0.005 a 0.5
qt_discs = 2
num_samples = 2
bottom_limit = 5
top_limit = 25
possible_mass = np.zeros([num_samples, qt_discs])
for i in range(num_samples):
    possible_mass[i] = [
        np.random.uniform(bottom_limit, top_limit),
        np.random.uniform(bottom_limit, top_limit),
    ]

## possíveis distâncias radiais para desbalanceamento
## 1 valores aleatório entre 0.068 a 0.068
qt_discs = 2
num_samples = 1
bottom_limit = 0.068
top_limit = 0.068
possible_radial_distances = np.zeros([num_samples, qt_discs])
for i in range(num_samples):
    possible_radial_distances[i] = [
        np.random.uniform(bottom_limit, top_limit),
        np.random.uniform(bottom_limit, top_limit),
    ]

## possíveis ângulos para desbalanceamento
## 2 valores 0 e 90 em rad
qt_discs = 2
possible_phi = [0, 1.5708] 
possible_phi_misalignement = [0, 1.5708] 
possible_angles_misalignement = [0, 1.5708] 

## nós de trinca
## verifiquei no plot do metamodelo e coloquei equidistante entre os discos, nó 19
possible_crack_nodes = [19]

## profundidades de trinca
## 20% e 40%
possible_crack_depth = [0.2, 0.4]

## possíveis delta rub
## valores entre 2 a 18 com passo de 5
## result = [0.1mm, 1mm e 2mm]
possible_delta_rub = [0.1/1000, 1/1000, 2/1000]

## possíveis posições de rub
## coloquei nos NOS 5 e 35
possible_pos_rub = [5, 35]

#############################################################################################################################################


probes = []
for probe in probes_nodes:
    for angle in probes_angles:
        probes.append((int(probe), angle))


def rubbing_simulation(debug=False):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("iniciando simulação: ")

    total_interaction = 10

    print("total de interações: ", total_interaction)

    objs = []
    aux = {}
    count_interaction = 0
    try:
        for speed in speeds:
            for mass in possible_unbalance_magnitudes:
                for phi in possible_unbalance_phases:
                    for delta_rub in possible_delta_rub:
                        for pos_rub in possible_pos_rub:
                            fig = sr.run(
                                dt,
                                tf,
                                speed,
                                mass,
                                phi,
                                probes,
                                delta_rub,
                                pos_rub, 
                            )
                            count_interaction += 1

                            print(
                                "progresso: "
                                + str(
                                    np.round(
                                        (count_interaction / total_interaction) * 100, 0
                                    )
                                )
                                + "% \r",
                                end="",
                            )
                            ############################
                            # Save
                            ############################
                            # get all the values of the array fig and truncate to 2 decimal places

                            for j, probe in enumerate(probes):
                                aux = {
                                    "probe": probe,
                                    "speed": speed,
                                    "magnitude": mass,
                                    "phase": phi,
                                    "delta_rub": delta_rub,
                                    "pos_rub": pos_rub,
                                    "y": fig.tolist()[-round(tf_desejado * (1 / dt)) :],
                                }
                                objs.append(aux)
                            if debug and count_interaction > limit_of_debug_samples:
                                print("parando simulação")
                                raise StopIteration
    except StopIteration as e:
        print("erro: ", e)
        pass

    df = pd.json_normalize(objs)
    df.to_parquet("results/rubbing.parquet")
    del df
    del objs


def misalignment_simulation(debug=False):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("iniciando simulação: ")

    total_interaction = 10

    print("total de interações: ", total_interaction)

    objs = []
    aux = {}
    count_interaction = 0
    try:
        for speed in speeds:
            for mass in possible_mass_misalignement:
                for phi in possible_phi_misalignement:
                    for misalignment_angle in possible_angles_misalignement:
                        fig = sm.run(
                            dt,
                            tf,
                            probes,
                            speed,
                            phi,
                            mass,
                            misalignment_angle,
                        )
                        count_interaction += 1
                        # progresso:
                        print(
                            "progresso: "
                            + str(
                                np.round(
                                    (count_interaction / total_interaction) * 100, 0
                                )
                            )
                            + "% \r",
                            end="",
                        )
                        ############################
                        # Save
                        ############################

                        for j, probe in enumerate(probes):
                            aux = {
                                "probe": probe,
                                "speed": speed,
                                "magnitude": mass,
                                "phase": phi,
                                "misalignment_angle": misalignment_angle,
                                "y": fig.tolist()[-round(tf_desejado * (1 / dt)) :],
                            }
                            objs.append(aux)
                        if debug and count_interaction > limit_of_debug_samples:
                            print("parando simulação")

                            raise StopIteration
    except StopIteration as e:
        print("erro: ", e)
        pass

    df = pd.json_normalize(objs)
    df.to_parquet("results/misalignment.parquet")
    del df
    del objs


def unbalance_simulation(debug=False):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("iniciando simulação: ")

    total_interaction = 10

    print("total de interações: ", total_interaction)

    objs = []
    aux = {}
    count_interaction = 0
    try:
        for speed in speeds:
            for mass in possible_unbalance_magnitudes:
                for phi in possible_unbalance_phases:
                        fig = smb.run(
                            dt,
                            tf,
                            probes,
                            speed,
                            mass, 
                            phi
                        )
                        count_interaction += 1
                        # progresso:
                        print(
                            "progresso: "
                            + str(
                                np.round(
                                    (count_interaction / total_interaction) * 100, 0
                                )
                            )
                            + "% \r",
                            end="",
                        )
                        ############################
                        # Save
                        ############################

                        for j, probe in enumerate(probes):
                            aux = {
                                "probe": probe,
                                "speed": speed,
                                "mass": mass,
                                "phi": phi,
                                "y": fig.tolist()[-round(tf_desejado * (1 / dt)) :],
                            }
                            objs.append(aux)
                        if debug and count_interaction > limit_of_debug_samples:
                            print("parando simulação")
                            raise StopIteration

    except StopIteration as e:
        print("erro: ", e)
        pass

    df = pd.json_normalize(objs)
    df.to_parquet("results/unbalance.parquet")
    del df
    del objs


def crack_simulation(debug=False):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("iniciando simulação: ")

    total_interaction = 10

    print("total de interações: ", total_interaction)

    objs = []
    aux = {}
    count_interaction = 0
    try:
        for speed in speeds:
            for mass in possible_unbalance_magnitudes:
                for phi in possible_unbalance_phases:
                        for crack in possible_crack_nodes:
                            for depth in possible_crack_depth:
                                print(
                                    "speed: "
                                    + str(speed)
                                    + " mass: "
                                    + str(mass)
                                    + " phi: "
                                    + str(phi)
                                    + " crack: "
                                    + str(crack)
                                    + " depth: "
                                    + str(depth)
                                )
                                fig = sc.run(
                                    dt,
                                    tf,
                                    probes,
                                    speed,
                                    mass,
                                    phi,
                                    int(crack),
                                    depth,
                                )
                                count_interaction += 1
                                # progresso:
                                print(
                                    "progresso: "
                                    + str(
                                        np.round(
                                            (count_interaction / total_interaction) * 100,
                                            0,
                                        )
                                    )
                                    + "% \r",
                                    end="",
                                )
                                ############################
                                # Save
                                ############################

                                for j, probe in enumerate(probes):
                                    aux = {
                                        "probe": probe,
                                        "speed": speed,
                                        "mass": mass,
                                        "phi": phi,
                                        "crack": crack,
                                        "depth": depth,
                                        "y": fig.tolist()[-round(tf_desejado * (1 / dt)) :],
                                    }
                                    objs.append(aux)
                                if debug and count_interaction > limit_of_debug_samples:
                                    print("parando simulação")

                                    raise StopIteration
    except StopIteration as e:
        print("erro: ", e)
        pass

    df = pd.json_normalize(objs)
    df.to_parquet("results/crack.parquet")
    del df
    del objs

def normal_simulation(debug=False):
    possible_normal_magnitudes = [((5/1000000)*dist)]
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("iniciando simulação: ")

    total_interaction = 10

    print("total de interações: ", total_interaction)

    objs = []
    aux = {}
    count_interaction = 0
    try:
        for speed in speeds:
            for mass in possible_normal_magnitudes:
                for phi in possible_unbalance_phases:
                        fig = smb.run(
                            dt,
                            tf,
                            probes,
                            speed,
                            mass, 
                            phi
                        )
                        count_interaction += 1
                        # progresso:
                        print(
                            "progresso: "
                            + str(
                                np.round(
                                    (count_interaction / total_interaction) * 100, 0
                                )
                            )
                            + "% \r",
                            end="",
                        )
                        ############################
                        # Save
                        ############################

                        for j, probe in enumerate(probes):
                            aux = {
                                "probe": probe,
                                "speed": speed,
                                "mass": mass,
                                "phi": phi,
                                "y": fig.tolist()[-round(tf_desejado * (1 / dt)) :],
                            }
                            objs.append(aux)
                        if debug and count_interaction > limit_of_debug_samples:
                            print("parando simulação")
                            raise StopIteration

    except StopIteration as e:
        print("erro: ", e)
        pass

    df = pd.json_normalize(objs)
    df.to_parquet("results/normal.parquet")
    del df
    del objs

if __name__ == "__main__":
    FLAG_DEBUG = True
    print("iniciando simulações")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # print("rubbing_simulation")
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # rubbing_simulation(FLAG_DEBUG) ## DEU CERTO

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # print("unbalance_simulation")
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # unbalance_simulation(FLAG_DEBUG) ## DEU CERTO

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # print("crack_simulation")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # crack_simulation(FLAG_DEBUG) ## DEU CERTO

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # print("misalignment_simulation")
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # misalignment_simulation(FLAG_DEBUG) ## DEU CERTO

    print("normal_simulation")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    normal_simulation(FLAG_DEBUG) 

    print("fim das simulações")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
