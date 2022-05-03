import mediapipe as mp
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from hand_tracking.tracking import mp_hands

matplotlib.use("GTK3Agg")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_hand_lms(img, hand):
	# img.flags.writeable = True
	mp_drawing.draw_landmarks(
		img,
		hand["lms_media_pipe"],
		mp_hands.HAND_CONNECTIONS,
		mp_drawing_styles.get_default_hand_landmarks_style(),
		mp_drawing_styles.get_default_hand_connections_style())
	return img

class Pose3DViewer:

	def __init__(self, fps: float=50) -> None:
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111, projection="3d")
		# self.ax.set_xlim(-0.7, 0.5)
		self.ax.set_xlim(0, 1)
		self.ax.set_ylim(0, 1)
		self.ax.set_zlim(0, 1)
		self.ax.set_xlabel("x")
		self.ax.set_ylabel("y")
		self.ax.set_zlabel("z")
		self.fig.tight_layout()
		self.ax.view_init(azim=-176.62037926349313, elev=21.890985171476757)
		self.update_time = 1/fps
		plt.ion()

	def plot_pose(self, pose, hand: dict=None) -> None:
		position = pose[0:3, 3]
		self.x_axis = self.ax.quiver(*position, *pose[0:3, 0], length=0.17, color="r")
		self.y_axis = self.ax.quiver(*position, *pose[0:3, 1], length=0.17, color="g")
		self.z_axis = self.ax.quiver(*position, *pose[0:3, 2], length=0.17, color="b")

		if hand:
			hand_lms_world = hand["lms_world"] + position # offset to match current hand position in world wrt. hip position
			self.scatter = self.ax.scatter(hand_lms_world[:, 0], hand_lms_world[:, 1], hand_lms_world[:, 2], linewidths=3, color="grey")

		plt.pause(self.update_time)
		# print(f"ax.azim {self.ax.azim}")
		# print(f"ax.elev {self.ax.elev}")

		if hand:
			self.scatter.remove()
		self.x_axis.remove()
		self.y_axis.remove()
		self.z_axis.remove()