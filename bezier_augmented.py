import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
from PIL import Image
import math

class InteractiveWettabilityCalculator:
    def __init__(self, master, image_path):
        self.master = master
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Load and display the image
        self.img = Image.open(image_path)
        self.img_array = np.array(self.img)
        self.ax.imshow(self.img_array)
        
        # Set the aspect of the plot to match the image
        self.ax.set_aspect('equal')
        
        # Initialize points
        height, width = self.img_array.shape[:2]
        self.p0 = np.array([width * 0.2, height * 0.8])
        self.p1 = np.array([width * 0.5, height * 0.5])
        self.p2 = np.array([width * 0.8, height * 0.8])
        
        self.selected_point = None
        
        # Plot elements
        self.surface_line, = self.ax.plot([], [], 'g--', linewidth=2, label='Surface')
        self.tangent_line, = self.ax.plot([], [], 'r--', linewidth=2, label='Tangent')
        self.control_line, = self.ax.plot([], [], 'b--', alpha=0.5)
        self.points, = self.ax.plot([], [], 'ro', markersize=10, picker=5)
        
        self.angle_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=10, 
                                       verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        self.ax.set_title('Ajouter des points en cliquant sur l\'image')
        self.ax.legend()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.update()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        for i, point in enumerate([self.p0, self.p1, self.p2]):
            if np.sqrt((event.xdata - point[0])**2 + (event.ydata - point[1])**2) < 10:
                self.selected_point = i
                break

    def on_release(self, event):
        self.selected_point = None

    def on_motion(self, event):
        if self.selected_point is None or event.inaxes != self.ax:
            return
        if self.selected_point == 0:
            self.p0 = np.array([event.xdata, event.ydata])
        elif self.selected_point == 1:
            self.p1 = np.array([event.xdata, event.ydata])
        else:
            self.p2 = np.array([event.xdata, event.ydata])
        self.update()

    def calculate_angle(self):
        # Vector from P0 to P1 (this is the direction of the tangent at P0)
        tangent_vector = self.p1 - self.p0
        
        # Normalize the tangent vector
        tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
        
        # Surface vector (from P0 to P2)
        surface_vector = self.p2 - self.p0
        
        # Normalize the surface vector
        surface_vector = surface_vector / np.linalg.norm(surface_vector)
        
        # Calculate the angle between the tangent and the surface
        dot_product = np.dot(tangent_vector, surface_vector)
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        
        # Ensure we're measuring the inner angle
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        
        return angle_deg

    def update(self):
        # Update surface line
        self.surface_line.set_data([self.p0[0], self.p2[0]], [self.p0[1], self.p2[1]])
        
        # Update control line
        self.control_line.set_data([self.p0[0], self.p1[0], self.p2[0]], [self.p0[1], self.p1[1], self.p2[1]])
        
        # Update points
        self.points.set_data([self.p0[0], self.p1[0], self.p2[0]], [self.p0[1], self.p1[1], self.p2[1]])
        
        # Calculate and update angle
        angle = self.calculate_angle()
        self.angle_text.set_text(f' Angle: {angle:.2f}°')
        
        # Update tangent line
        tangent_vector = self.p1 - self.p0
        tangent_length = 100  # Adjust this for desired tangent line length
        tangent_end = self.p0 + tangent_vector / np.linalg.norm(tangent_vector) * tangent_length
        self.tangent_line.set_data([self.p0[0], tangent_end[0]], [self.p0[1], tangent_end[1]])
        
        self.canvas.draw()

root = tk.Tk()
root.title("Mouillabilité interactive")
app = InteractiveWettabilityCalculator(root, r'C:\Users\yassine\Desktop\prj\capture.jpg') 
root.mainloop()