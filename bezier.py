import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import symbols, expand

class InteractiveBezier:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.p0 = np.array([1, 1])
        self.p1 = np.array([4, 5])
        self.p2 = np.array([7, 3])
        self.selected_point = None
        self.t = np.linspace(0, 1, 100)
        
        self.curve_line, = self.ax.plot([], [], 'b-', label='surface bulle')
        self.control_line, = self.ax.plot([], [], 'r--', alpha=0.5)
        self.surface_line, = self.ax.plot([], [], 'g--', label='Surface du materiaux')
        self.tangent_line, = self.ax.plot([], [], 'm--', label='Tangente')
        self.points, = self.ax.plot([], [], 'ro', markersize=10, picker=5)
        
        self.angle_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, verticalalignment='top')
        self.poly_text = self.ax.text(0.05, 0.80, '', transform=self.ax.transAxes, verticalalignment='top', fontsize=9)
        
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_title('courbe de bezier interactive')
        self.ax.legend()
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.ani = FuncAnimation(self.fig, self.update, frames=range(100), interval=50, blit=True)

    def quadratic_bezier(self, t):
        t = t[:, np.newaxis]  # (100,) -> (100, 1)
        return (1-t)**2 * self.p0 + 2*(1-t)*t * self.p1 + t**2 * self.p2

    def bezier_tangent(self, t):
        return 2*(1-t)*(self.p1-self.p0) + 2*t*(self.p2-self.p1)

    def contact_angle(self):
        tangent = self.bezier_tangent(0)
        angle = np.arctan2(tangent[1], tangent[0])
        return np.degrees(angle)

    def bezier_to_polynomial(self):
        t = symbols('t')
        x = (1-t)**2 * self.p0[0] + 2*(1-t)*t * self.p1[0] + t**2 * self.p2[0]
        y = (1-t)**2 * self.p0[1] + 2*(1-t)*t * self.p1[1] + t**2 * self.p2[1]
        x_poly = expand(x)
        y_poly = expand(y)
        return f"x(t) = {x_poly}\ny(t) = {y_poly}"

    def update(self, frame):
        curve = self.quadratic_bezier(self.t)
        self.curve_line.set_data(curve[:, 0], curve[:, 1])
        self.control_line.set_data([self.p0[0], self.p1[0], self.p2[0]], [self.p0[1], self.p1[1], self.p2[1]])
        self.surface_line.set_data([self.p0[0], self.p0[0] + 2], [self.p0[1], self.p0[1]])
        
        angle = self.contact_angle()
        tangent_end = self.p0 + np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        self.tangent_line.set_data([self.p0[0], tangent_end[0]], [self.p0[1], tangent_end[1]])
        
        self.points.set_data([self.p0[0], self.p1[0], self.p2[0]], [self.p0[1], self.p1[1], self.p2[1]])
        
        self.angle_text.set_text(f' Angle de contact: {angle:.2f}°')
        self.poly_text.set_text(f'Expression du polynome:\n{self.bezier_to_polynomial()}')
        
        return self.curve_line, self.control_line, self.surface_line, self.tangent_line, self.points, self.angle_text, self.poly_text

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        for i, point in enumerate([self.p0, self.p1, self.p2]):
            if np.sqrt((event.xdata - point[0])**2 + (event.ydata - point[1])**2) < 0.1:
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

if __name__ == '__main__':
    interactive_bezier = InteractiveBezier()
    plt.show()