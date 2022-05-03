import numpy as np
from scipy.spatial.transform import Rotation

# R = Rotation.from_rotvec([0, np.pi/2, 0]).as_matrix() @ Rotation.from_rotvec([0, 0, np.pi/2]).as_matrix()
# R_eul = Rotation.from_euler("yxz", [-np.pi/2, -np.pi/2, 0]).as_matrix()


# print(f"R: {R}")
# print(f"R_eul: {R_eul}")
MIN = np.array([-0.4833117425441742, 0.05090602, 0.05353609])
MAX = np.array([0.1480371505022049,  0.7783022,  0.67509466])



offset = np.abs(MIN)
scale_factor = 1/(MAX + offset)
scaled = (offset + MAX)/(MAX + offset)
print(scaled)
print(f"scale_Factor: {scale_factor}")
print(f"scaled_by_factor {(MIN + offset) * scale_factor}")