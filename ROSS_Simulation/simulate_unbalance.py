import ross as rs
from utils import change_model_4dof_to_6dof
from ross.units import Q_, check_units
import numpy as np

rotor_file = "lmest_bancada_2discos.toml"


# # Unbalance parameters of [disk1, disk2, ...]
# mass = [0.005, 0.5]
# d = [0.068, 0.3]
# phi = [70, 30]
def run(dt, tf, probes, speed, unbalance_magnitude, unbalance_phase):
    ############################
    # Input
    ############################

    magunbt = np.array([unbalance_magnitude, unbalance_magnitude])
    phaseunbt = np.array([unbalance_phase, unbalance_phase])

    rotor = rs.Rotor.load(rotor_file)
    # alpha and beta are the proportional damping coefficients
    rotor = change_model_4dof_to_6dof(rotor, alpha=1, beta=1e-4)

    ############################
    # Response
    ############################
    misalignment = rotor.run_misalignment(
        coupling="flex",
        dt=dt,
        tI=0,
        tF=tf,
        kd=1e-6,
        ks=1e-6,
        eCOUPx=1e-6,
        eCOUPy=1e-6,
        misalignment_angle=0,
        TD=0,
        TL=0,
        n1=0,
        speed=Q_(speed, "RPM"),
        unbalance_magnitude=magunbt,
        unbalance_phase=phaseunbt,
        mis_type="parallel",
        print_progress=False,
    )

    fig, sinal_normal_recons = misalignment.plot_dfft(probe=probes, yaxis_type="log")

    return sinal_normal_recons
