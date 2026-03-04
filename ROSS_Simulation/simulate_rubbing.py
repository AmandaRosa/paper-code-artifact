import pandas as pd
import ross as rs
import numpy as np
from utils import change_model_4dof_to_6dof
from ross.units import Q_, check_units
from ross.defects.rubbing import rubbing_example

rotor_file = "lmest_bancada_2discos.toml"


def run(
    dt,
    tf,
    speed,
    unbalance_magnitude,
    unbalance_phase,
    probes,
    deltaRub,
    posRub,
    kRub=1.1e6,
    cRub=40,
    miRub=0.3,
):
    rotor = rs.Rotor.load(rotor_file)

    magunbt = np.array([unbalance_magnitude, unbalance_magnitude])
    phaseunbt = np.array([unbalance_phase, unbalance_phase])

    # alpha and beta are the proportional damping coefficients
    rotor = change_model_4dof_to_6dof(rotor, alpha=1, beta=1e-4)


    rubbing = rotor.run_rubbing(
        dt=dt,
        tI=0,
        tF=tf,
        deltaRUB=deltaRub,
        kRUB=kRub,
        cRUB=cRub,
        miRUB=miRub,
        posRUB=int(posRub),
        speed=Q_(speed, "RPM"),
        unbalance_magnitude=magunbt,
        unbalance_phase=phaseunbt,
        torque=False,
        print_progress=False,
    )

    # results = rubbing.run_time_response()
    # fig_time = results.plot_1d(probes)
    fig, sinal_normal_recons = rubbing.plot_dfft(probe=probes, yaxis_type="log")

    return sinal_normal_recons
