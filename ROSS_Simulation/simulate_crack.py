import ross as rs
from ross.units import Q_, check_units
import numpy as np
from utils import change_model_4dof_to_6dof

rotor_file = "lmest_bancada_2discos.toml"


def run(dt, tf, probes, speed, mass, phi, crack_node, crack_depth_ratio):
   
    phaseunbt = np.array([phi, phi]) 

    magunbt = np.array([mass, mass])

    rotor = rs.Rotor.load(rotor_file)

    rotor = change_model_4dof_to_6dof(rotor, alpha=1, beta=1e-4)

    crack = rotor.run_crack(
        dt=dt,
        tI=0.0,
        tF=tf,
        depth_ratio=crack_depth_ratio,
        n_crack=crack_node,
        speed=Q_(speed, "RPM"),
        unbalance_magnitude=magunbt,
        unbalance_phase=phaseunbt,
        crack_type="Mayes",
        print_progress=False,
    )

    # time_response = crack.run_time_response()

    # fig_time = time_response.plot_1d(probes)
    fig, sinal_normal_recons = crack.plot_dfft(probe=probes, yaxis_type="log")
    # fig.show()
    return sinal_normal_recons


# run(
#     dt=4e-5,
#     tf=8,
#     probes=[(4, 1.2566370614359172)],
#     speed=1000,
#     mass=[0.5, 0.5],
#     d=[0.10, 0.23],
#     phi=[0.068, 0.068],
#     crack_node=8,
#     crack_depth_ratio=0.5,
# )
