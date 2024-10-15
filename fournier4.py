import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSlider, 
                             QComboBox, QLabel, QPushButton, QLineEdit, QGridLayout, QTabWidget, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtGui import QImage

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FunctionGrapher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.setWindowTitle("Comprehensive Function Grapher")
        self.setGeometry(100, 100, 1600, 900)

        self.usual_functions = {
            "Linear": lambda x, a, b, c: (a * x + b, f"y = {a:.2f}x + {b:.2f}"),
            "Quadratic": lambda x, a, b, c: (a * x**2 + b * x + c, f"y = {a:.2f}x² + {b:.2f}x + {c:.2f}"),
            "Cubic": lambda x, a, b, c: (a * x**3 + b * x**2 + c * x, f"y = {a:.2f}x³ + {b:.2f}x² + {c:.2f}x"),
            "Sine": lambda x, a, b, c: (a * np.sin(b * x + c), f"y = {a:.2f}sin({b:.2f}x + {c:.2f})"),
            "Cosine": lambda x, a, b, c: (a * np.cos(b * x + c), f"y = {a:.2f}cos({b:.2f}x + {c:.2f})"),
            "Exponential": lambda x, a, b, c: (a * np.exp(b * x) + c, f"y = {a:.2f}e^({b:.2f}x) + {c:.2f}"),
            "Logarithmic": lambda x, a, b, c: (a * np.log(np.abs(b * x)) + c, f"y = {a:.2f}ln(|{b:.2f}x|) + {c:.2f}")
        }

        self.signal_functions = {
            "Square Wave": (self.square_wave, self.calculate_square_wave),
            "Triangle Wave": (self.triangle_wave, self.calculate_triangle_wave),
            "Sawtooth Wave": (self.sawtooth_wave, self.calculate_sawtooth_wave),
            "Arche Signal": (self.arche_signal, self.calculate_arche_signals)
        }

        self.probability_distributions = {
            "Normal": stats.norm,
            "Uniform": stats.uniform,
            "Exponential": stats.expon,
            "Poisson": stats.poisson,
            "Binomial": stats.binom
        }

        self.distribution_params = {
            "Normal": ("μ (mean)", "σ (std dev)"),
            "Uniform": ("a (min)", "b (max)"),
            "Exponential": ("λ (rate)", ""),
            "Poisson": ("λ (mean)", ""),
            "Binomial": ("n (trials)", "p (probability)")
        }

        self.setup_ui()

        # Load MNIST model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mnist_model = Net().to(self.device)
        self.mnist_model.load_state_dict(torch.load('mnist_cnn_single_model.pth', map_location=self.device))
        self.mnist_model.eval()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Usual Functions Tab
        usual_tab = QWidget()
        usual_layout = QVBoxLayout(usual_tab)
        self.setup_usual_functions(usual_layout)
        self.tab_widget.addTab(usual_tab, "Usual Functions")

        # Signal Analysis Tab
        signal_tab = QWidget()
        signal_layout = QVBoxLayout(signal_tab)
        self.setup_signal_analysis(signal_layout)
        self.tab_widget.addTab(signal_tab, "Signal Analysis")

        # 3D Signal Analysis Tab
        signal_3d_tab = QWidget()
        signal_3d_layout = QVBoxLayout(signal_3d_tab)
        self.setup_3d_signal_analysis(signal_3d_layout)
        self.tab_widget.addTab(signal_3d_tab, "3D Signal Analysis")

        # Statistical Analysis Tab
        stat_tab = QWidget()
        stat_layout = QVBoxLayout(stat_tab)
        self.setup_statistical_analysis(stat_layout)
        self.tab_widget.addTab(stat_tab, "Statistical Analysis")

        # MNIST Digit Recognition Tab
        mnist_tab = QWidget()
        mnist_layout = QVBoxLayout(mnist_tab)
        self.setup_mnist_recognition(mnist_layout)
        self.tab_widget.addTab(mnist_tab, "MNIST Digit Recognition")

        # Image Processing Tab
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        self.setup_image_processing(image_layout)
        self.tab_widget.addTab(image_tab, "Image Processing")

    def setup_image_processing(self, layout):
        # Image selection
        image_layout = QHBoxLayout()
        self.image_path = QLineEdit()
        self.image_button = QPushButton("Browse")
        self.image_button.clicked.connect(self.browse_image)
        image_layout.addWidget(QLabel("Select an image:"))
        image_layout.addWidget(self.image_path)
        image_layout.addWidget(self.image_button)
        layout.addLayout(image_layout)

        # Image display
        self.image_display = QLabel()
        self.image_display.setFixedSize(400, 400)
        self.image_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_display)

        # Processing options
        self.process_combo = QComboBox()
        self.process_combo.addItems(["Grayscale", "Blur", "Edge Detection"])
        layout.addWidget(QLabel("Select processing:"))
        layout.addWidget(self.process_combo)

        # Process button
        self.process_button = QPushButton("Process Image")
        self.process_button.clicked.connect(self.process_image)
        layout.addWidget(self.process_button)

        # Processed image display
        self.processed_display = QLabel()
        self.processed_display.setFixedSize(400, 400)
        self.processed_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.processed_display)

    def browse_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        print(f"Selected file: {file_name}")  # Debug print
        
        if file_name:
            if not os.path.exists(file_name):
                print(f"Error: File does not exist: {file_name}")
                return
            
            self.image_path.setText(file_name)
            print(f"Image path set to: {self.image_path.text()}")  # Debug print
            self.load_and_display_image(file_name)
        else:
            print("No file selected")

    def load_and_display_image(self, file_name):
        print(f"Attempting to load image from: {file_name}")  # Debug print
        self.original_image = cv2.imread(file_name)
        
        if self.original_image is None:
            print(f"Error: Unable to load image from {file_name}")
            return

        print(f"Image loaded successfully. Shape: {self.original_image.shape}")  # Debug print

        # Convert to RGB for display
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Display in MNIST tab
        self.image_display.setPixmap(pixmap.scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        print("Image displayed in MNIST tab")  # Debug print

        # Display in Image Processing tab
        if hasattr(self, 'image_processing_display'):
            self.image_processing_display.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            print("Image displayed in Image Processing tab")  # Debug print
        else:
            print("Warning: image_processing_display not found")  # Debug print
        
        self.processed_display.clear() 

    def update_image_processing_display(self, pixmap):
        # Assuming you have a QLabel in the Image Processing tab to display the image
        if hasattr(self, 'image_processing_display'):
            self.image_processing_display.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            print("Image displayed in Image Processing tab")  # Debug print
        else:
            print("Warning: image_processing_display not found")  # Debug print
    

    def process_image(self):
        if not self.image_path.text():
            return

        image = cv2.imread(self.image_path.text())
        process = self.process_combo.currentText()

        if process == "Grayscale":
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif process == "Blur":
            processed = cv2.GaussianBlur(image, (15, 15), 0)
        elif process == "Edge Detection":
            processed = cv2.Canny(image, 100, 200)

        height, width = processed.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(processed.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.processed_display.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def setup_usual_functions(self, layout):
        control_layout = QHBoxLayout()
        
        self.usual_function_combo = QComboBox()
        self.usual_function_combo.addItems(self.usual_functions.keys())
        self.usual_function_combo.currentIndexChanged.connect(self.update_usual_plot)
        control_layout.addWidget(QLabel("Function:"))
        control_layout.addWidget(self.usual_function_combo)

        self.usual_param_sliders = []
        for i in range(3):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(0)
            slider.setTickInterval(20)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.valueChanged.connect(self.update_usual_plot)
            self.usual_param_sliders.append(slider)
            control_layout.addWidget(QLabel(f"Param {i+1}:"))
            control_layout.addWidget(slider)

        layout.addLayout(control_layout)

        self.usual_figure = Figure(figsize=(10, 6))
        self.usual_canvas = FigureCanvas(self.usual_figure)
        layout.addWidget(self.usual_canvas)

        self.update_usual_plot()

    def setup_signal_analysis(self, layout):
        control_layout = QHBoxLayout()
        
        self.signal_function_combo = QComboBox()
        self.signal_function_combo.addItems(self.signal_functions.keys())
        control_layout.addWidget(QLabel("Function:"))
        control_layout.addWidget(self.signal_function_combo)

        self.H_input = QLineEdit("10")
        self.a_input = QLineEdit("1")
        self.f_input = QLineEdit("1")
        self.fe_input = QLineEdit("1000")
        self.fin_input = QLineEdit("1")
        
        for label, widget in [("H:", self.H_input), ("a:", self.a_input), ("f:", self.f_input), 
                            ("fe:", self.fe_input), ("fin:", self.fin_input)]:
            control_layout.addWidget(QLabel(label))
            control_layout.addWidget(widget)

        # Add Gaussian noise controls
        self.noise_checkbox = QCheckBox("Add Gaussian Noise")
        self.noise_checkbox.stateChanged.connect(self.toggle_noise_controls)
        control_layout.addWidget(self.noise_checkbox)

        self.noise_mean_input = QLineEdit("0")
        self.noise_std_input = QLineEdit("0.1")
        self.noise_mean_input.setEnabled(False)
        self.noise_std_input.setEnabled(False)
        control_layout.addWidget(QLabel("Noise Mean:"))
        control_layout.addWidget(self.noise_mean_input)
        control_layout.addWidget(QLabel("Noise Std:"))
        control_layout.addWidget(self.noise_std_input)

        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self.update_signal_plot)
        control_layout.addWidget(self.apply_button)

        layout.addLayout(control_layout)

        plots_layout = QGridLayout()
        self.signal_figures = []
        self.signal_canvases = []

        titles = [
            "Theoretical Signal",
            "Harmonics",
            "2D Harmonics Representation",
            "Reconstructed Signal",
            "FFT Estimation",
            "Weighted Harmonics"
        ]

        for i in range(6):
            fig = Figure(figsize=(5, 4))
            canvas = FigureCanvas(fig)
            self.signal_figures.append(fig)
            self.signal_canvases.append(canvas)
            plots_layout.addWidget(canvas, i // 3, i % 3)
            ax = fig.add_subplot(111)
            ax.set_title(titles[i])

        layout.addLayout(plots_layout)

    def toggle_noise_controls(self, state):
        self.noise_mean_input.setEnabled(state == Qt.Checked)
        self.noise_std_input.setEnabled(state == Qt.Checked)

    def setup_3d_signal_analysis(self, layout):
        control_layout = QHBoxLayout()
        
        self.signal_3d_function_combo = QComboBox()
        self.signal_3d_function_combo.addItems(self.signal_functions.keys())
        control_layout.addWidget(QLabel("Function:"))
        control_layout.addWidget(self.signal_3d_function_combo)

        self.apply_3d_button = QPushButton("Apply Changes")
        self.apply_3d_button.clicked.connect(self.update_3d_signal_plot)
        control_layout.addWidget(self.apply_3d_button)

        layout.addLayout(control_layout)

        plots_layout = QGridLayout()
        self.signal_3d_figures = []
        self.signal_3d_canvases = []

        titles = [
            "3D Fourier Decomposition",
            "3D Weighted Harmonics"
        ]

        for i in range(2):
            fig = Figure(figsize=(8, 6))
            canvas = FigureCanvas(fig)
            self.signal_3d_figures.append(fig)
            self.signal_3d_canvases.append(canvas)
            plots_layout.addWidget(canvas, 0, i)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(titles[i])

        layout.addLayout(plots_layout)

    def setup_statistical_analysis(self, layout):
        control_layout = QHBoxLayout()
        
        self.dist_combo = QComboBox()
        self.dist_combo.addItems(self.probability_distributions.keys())
        self.dist_combo.currentIndexChanged.connect(self.update_param_labels)
        control_layout.addWidget(QLabel("Distribution:"))
        control_layout.addWidget(self.dist_combo)

        self.param1_label = QLabel("Param 1:")
        self.param1_input = QLineEdit("0")
        self.param2_label = QLabel("Param 2:")
        self.param2_input = QLineEdit("1")
        control_layout.addWidget(self.param1_label)
        control_layout.addWidget(self.param1_input)
        control_layout.addWidget(self.param2_label)
        control_layout.addWidget(self.param2_input)

        self.sample_size_input = QLineEdit("1000")
        control_layout.addWidget(QLabel("Sample Size:"))
        control_layout.addWidget(self.sample_size_input)

        self.apply_stat_button = QPushButton("Generate Distribution")
        self.apply_stat_button.clicked.connect(self.update_stat_plot)
        control_layout.addWidget(self.apply_stat_button)

        self.apply_clt_button = QPushButton("Apply CLT")
        self.apply_clt_button.clicked.connect(self.apply_clt)
        control_layout.addWidget(self.apply_clt_button)

        layout.addLayout(control_layout)

        self.stat_figure = Figure(figsize=(10, 6))
        self.stat_canvas = FigureCanvas(self.stat_figure)
        layout.addWidget(self.stat_canvas)

        self.update_param_labels()

    def setup_mnist_recognition(self, layout):
        # Image selection
        image_layout = QHBoxLayout()
        self.image_label = QLabel("Select an image:")
        self.image_path = QLineEdit()
        self.image_button = QPushButton("Browse")
        self.image_button.clicked.connect(self.browse_image)
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.image_path)
        image_layout.addWidget(self.image_button)
        layout.addLayout(image_layout)

        # Recognize button
        self.recognize_button = QPushButton("Recognize Digit")
        self.recognize_button.clicked.connect(self.recognize_digit)
        layout.addWidget(self.recognize_button)

        # Results display
        results_layout = QHBoxLayout()
        self.image_display = QLabel()
        self.image_display.setFixedSize(280, 280)
        self.image_display.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.image_display)

        self.mnist_figure = Figure(figsize=(5, 4))
        self.mnist_canvas = FigureCanvas(self.mnist_figure)
        results_layout.addWidget(self.mnist_canvas)

        layout.addLayout(results_layout)

        # Prediction result
        self.prediction_label = QLabel()
        layout.addWidget(self.prediction_label)

    def update_param_labels(self):
        dist_name = self.dist_combo.currentText()
        param1_name, param2_name = self.distribution_params[dist_name]
        self.param1_label.setText(f"{param1_name}:")
        self.param2_label.setText(f"{param2_name}:")
        self.param2_label.setVisible(bool(param2_name))
        self.param2_input.setVisible(bool(param2_name))

    def update_usual_plot(self):
        function_name = self.usual_function_combo.currentText()
        x = np.linspace(-10, 10, 1000)
        params = [slider.value() / 20 for slider in self.usual_param_sliders]
        
        y, title = self.usual_functions[function_name](x, *params)

        self.usual_figure.clear()
        ax = self.usual_figure.add_subplot(111)
        ax.plot(x, y)
        ax.set_title(title)
        ax.grid(True)
        self.usual_canvas.draw()

    def update_signal_plot(self):
        try:
            function_name = self.signal_function_combo.currentText()
            H = int(self.H_input.text())
            a = float(self.a_input.text())
            f = float(self.f_input.text())
            fe = float(self.fe_input.text())
            fin = float(self.fin_input.text())

            if fe == 0:
                raise ValueError("Sampling frequency (fe) cannot be zero.")

            t = np.arange(0, fin, 1/fe)

            theoretical_func, fourier_func = self.signal_functions[function_name]
            theoretical_signal = theoretical_func(t, a, f)
            TabConcat, TabSignal, TabReconstr, reconstruction = fourier_func(H, a, f, t)

            # Add Gaussian noise if checkbox is checked
            if self.noise_checkbox.isChecked():
                noise_mean = float(self.noise_mean_input.text())
                noise_std = float(self.noise_std_input.text())
                noise = np.random.normal(noise_mean, noise_std, theoretical_signal.shape)
                theoretical_signal += noise
                reconstruction += noise

            self.plot_signal_results(theoretical_signal, TabConcat, TabSignal, TabReconstr, reconstruction, 0, len(t)-1, function_name)
        except ValueError as e:
            print(f"Error: {str(e)}. Please enter valid numeric values for all parameters.")

    def update_3d_signal_plot(self):
        try:
            function_name = self.signal_3d_function_combo.currentText()
            H = int(self.H_input.text())
            a = float(self.a_input.text())
            f = float(self.f_input.text())
            fe = float(self.fe_input.text())
            fin = float(self.fin_input.text())

            if fe == 0:
                raise ValueError("Sampling frequency (fe) cannot be zero.")

            t = np.arange(0, fin, 1/fe)

            _, fourier_func = self.signal_functions[function_name]
            TabConcat, TabSignal, TabReconstr, reconstruction = fourier_func(H, a, f, t)

            self.plot_3d_signal_results(TabReconstr, TabSignal, function_name)
        except ValueError as e:
            print(f"Error: {str(e)}. Please enter valid numeric values for all parameters.")

    def update_stat_plot(self):
        try:
            dist_name = self.dist_combo.currentText()
            param1 = float(self.param1_input.text())
            param2 = float(self.param2_input.text()) if self.param2_input.isVisible() else None
            sample_size = int(self.sample_size_input.text())

            if sample_size <= 0:
                raise ValueError("Sample size must be positive.")

            dist = self.probability_distributions[dist_name]
            
            if dist_name == "Normal":
                data = dist.rvs(loc=param1, scale=param2, size=sample_size)
                pdf = lambda x: dist.pdf(x, loc=param1, scale=param2)
            elif dist_name == "Uniform":
                if param1 >= param2:
                    raise ValueError("For Uniform distribution, 'a' must be less than 'b'.")
                data = dist.rvs(loc=param1, scale=param2-param1, size=sample_size)
                pdf = lambda x: dist.pdf(x, loc=param1, scale=param2-param1)
            elif dist_name == "Exponential":
                if param1 <= 0:
                    raise ValueError("For Exponential distribution, λ (rate) must be positive.")
                data = dist.rvs(scale=1/param1, size=sample_size)
                pdf = lambda x: dist.pdf(x, scale=1/param1)
            elif dist_name == "Poisson":
                if param1 <= 0:
                    raise ValueError("For Poisson distribution, λ (mean) must be positive.")
                data = dist.rvs(mu=param1, size=sample_size)
                pdf = lambda x: dist.pmf(x, mu=param1)
            elif dist_name == "Binomial":
                n = int(param1)
                p = param2
                if n <= 0:
                    raise ValueError("For Binomial distribution, n (trials) must be a positive integer.")
                if not 0 <= p <= 1:
                    raise ValueError("For Binomial distribution, p (probability) must be between 0 and 1.")
                data = dist.rvs(n=n, p=p, size=sample_size)
                pdf = lambda x: dist.pmf(x, n=n, p=p)

            self.stat_figure.clear()
            ax = self.stat_figure.add_subplot(111)
            
            if dist_name in ["Poisson", "Binomial"]:
                unique, counts = np.unique(data, return_counts=True)
                ax.bar(unique, counts/sample_size, alpha=0.7, color='b', label='Data')
                x = np.arange(min(data), max(data)+1)
            else:
                ax.hist(data, bins=50, density=True, alpha=0.7, color='b', label='Data')
                x = np.linspace(min(data), max(data), 100)
            
            ax.plot(x, [pdf(xi) for xi in x], 'r-', lw=2, label='PDF/PMF')
            
            ax.set_title(f"{dist_name} Distribution")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density" if dist_name not in ["Poisson", "Binomial"] else "Probability")
            ax.legend()
            self.stat_canvas.draw()

        except ValueError as e:
            print(f"Error: {str(e)}. Please enter valid numeric values for all parameters.")

    def apply_clt(self):
        try:
            dist_name = self.dist_combo.currentText()
            param1 = float(self.param1_input.text())
            param2 = float(self.param2_input.text())
            sample_size = int(self.sample_size_input.text())

            if sample_size <= 0:
                raise ValueError("Sample size must be positive.")

            dist = self.probability_distributions[dist_name]
            
            means = []
            for _ in range(1000):  # Generate 1000 sample means
                if dist_name == "Normal":
                    data = dist.rvs(loc=param1, scale=param2, size=sample_size)
                elif dist_name == "Uniform":
                    if param1 >= param2:
                        raise ValueError("For Uniform distribution, param1 (low) must be less than param2 (high).")
                    data = dist.rvs(loc=param1, scale=param2-param1, size=sample_size)
                elif dist_name == "Exponential":
                    if param1 <= 0:
                        raise ValueError("For Exponential distribution, param1 (rate) must be positive.")
                    data = dist.rvs(scale=1/param1, size=sample_size)
                elif dist_name == "Poisson":
                    if param1 <= 0:
                        raise ValueError("For Poisson distribution, param1 (mu) must be positive.")
                    data = dist.rvs(mu=param1, size=sample_size)
                elif dist_name == "Binomial":
                    n = int(param1)
                    p = param2
                    if n <= 0:
                        raise ValueError("For Binomial distribution, param1 (n) must be a positive integer.")
                    if not 0 <= p <= 1:
                        raise ValueError("For Binomial distribution, param2 (p) must be between 0 and 1.")
                    data = dist.rvs(n=n, p=p, size=sample_size)
                means.append(np.mean(data))

            self.stat_figure.clear()
            ax = self.stat_figure.add_subplot(111)
            ax.hist(means, bins=50, density=True, alpha=0.7, color='g', label='Sample Means')
            
            # Fit a normal distribution to the sample means
            mu, std = stats.norm.fit(means)
            x = np.linspace(min(means), max(means), 100)
            p = stats.norm.pdf(x, mu, std)
            ax.plot(x, p, 'r-', lw=2, label='Fitted Normal')
            
            ax.set_title(f"Central Limit Theorem - {dist_name} Distribution")
            ax.set_xlabel("Sample Mean")
            ax.set_ylabel("Density")
            ax.legend()
            self.stat_canvas.draw()

        except ValueError as e:
            print(f"Error: {str(e)}. Please enter valid numeric values for all parameters.")

    def plot_signal_results(self, theoretical_signal, TabConcat, TabSignal, TabReconstr, reconstruction, Xmin, Xmax, signal_name):
        TabAbsolu = np.abs(TabConcat[np.abs(TabConcat) >= 3.898171832519376e-17])
        
        plot_funcs = [
            lambda ax: ax.plot(theoretical_signal),
            lambda ax: (ax.stem(np.abs(TabAbsolu)), ax.plot(np.abs(TabAbsolu), 'r--')),
            lambda ax: ax.plot(TabReconstr.T),
            lambda ax: ax.plot(reconstruction),
            lambda ax: ax.stem(np.abs(np.fft.fft(reconstruction))[:60]),
            lambda ax: ax.plot(TabSignal.T)
        ]

        for fig, canvas, plot_func in zip(self.signal_figures, self.signal_canvases, plot_funcs):
            ax = fig.gca()
            ax.clear()
            plot_func(ax)
            ax.set_xlabel('Time (ms)' if ax.get_title() != 'Harmonics' else f'Coefficient of {signal_name}')
            ax.set_ylabel('Amplitude')
            canvas.draw()

    def plot_3d_signal_results(self, TabReconstr, TabSignal, signal_name):
        # 3D Fourier Decomposition
        ax = self.signal_3d_figures[0].gca()
        ax.clear()
        X, Y = np.meshgrid(range(TabReconstr.shape[1]), range(TabReconstr.shape[0]))
        ax.plot_surface(X, Y, TabReconstr, cmap='viridis')
        ax.set_title(f'3D Fourier Decomposition of {signal_name}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Harmonic Rank')
        ax.set_zlabel('Amplitude')
        self.signal_3d_canvases[0].draw()

        # 3D Weighted Harmonics
        ax = self.signal_3d_figures[1].gca()
        ax.clear()
        X, Y = np.meshgrid(range(TabSignal.shape[1]), range(TabSignal.shape[0]))
        ax.plot_surface(X, Y, TabSignal, cmap='viridis')
        ax.set_title(f'3D Weighted Harmonics of {signal_name}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Harmonic Rank')
        ax.set_zlabel('Amplitude')
        self.signal_3d_canvases[1].draw()

    # Theoretical waveform generators
    def square_wave(self, t, a, f):
        return a * np.sign(np.sin(2 * np.pi * f * t))

    def triangle_wave(self, t, a, f):
        return 2 * a / np.pi * np.arcsin(np.sin(2 * np.pi * f * t))

    def sawtooth_wave(self, t, a, f):
        return 2 * a * (f * t - np.floor(f * t + 0.5))

    def arche_signal(self, t, a, f):
        return a * (1 - 2 * np.abs(f * t - np.floor(f * t + 0.5)))

    # Fourier series calculation methods
    def calculate_square_wave(self, H, a, f, t):
        TabConcat = np.zeros(2*H)
        TabSignal = np.zeros((2*H, len(t)))

        for i in range(1, 2*H + 1):
            CoeffSquare = (2*a)/(np.pi*i)*np.sin(np.pi*i/2)
            TabConcat[i-1] = CoeffSquare

            HarmoSquare = np.cos(2*np.pi*i*f*t)
            TabSignal[i-1, :] = CoeffSquare * HarmoSquare

        TabReconstr = np.cumsum(TabSignal, axis=0)
        reconstruction = TabReconstr[-1, :]

        return TabConcat, TabSignal, TabReconstr, reconstruction

    def calculate_triangle_wave(self, H, a, f, t):
        TabConcat = np.zeros(2*H)
        TabSignal = np.zeros((2*H, len(t)))

        for i in range(1, 2*H + 1):
            CoeffTriangle = (-a)/(i**2)*np.sin((i*np.pi)/2)
            TabConcat[i-1] = CoeffTriangle

            HarmoTriangle = np.sin(2*np.pi*i*f*t)
            TabSignal[i-1, :] = CoeffTriangle * HarmoTriangle

        TabReconstr = np.cumsum(TabSignal, axis=0)
        reconstruction = TabReconstr[-1, :]

        return TabConcat, TabSignal, TabReconstr, reconstruction

    def calculate_sawtooth_wave(self, H, a, f, t):
        TabConcat = np.zeros(2*H)
        TabSignal = np.zeros((2*H, len(t)))

        for i in range(1, 2*H + 1):
            CoeffSawtooth = ((-a)/(2*np.pi*i))*np.cos(i*np.pi)
            TabConcat[i-1] = CoeffSawtooth

            HarmoSawtooth = np.sin(2*np.pi*i*f*t)
            TabSignal[i-1, :] = CoeffSawtooth * HarmoSawtooth

        TabReconstr = np.cumsum(TabSignal, axis=0)
        reconstruction = TabReconstr[-1, :]

        return TabConcat, TabSignal, TabReconstr, reconstruction

    def calculate_arche_signals(self, H, a, f, t):
        TabConcat = np.zeros(2*H)
        TabSignal = np.zeros((2*H, len(t)))

        for i in range(1, 2*H + 1):
            CoeffArche = -(1)/(i**2*np.pi)*(-1)**i
            TabConcat[i-1] = CoeffArche

            HarmoArche = np.cos(2*np.pi*i*f*t)
            TabSignal[i-1, :] = CoeffArche * HarmoArche

        TabReconstr = np.cumsum(TabSignal, axis=0)
        reconstruction = TabReconstr[-1, :]

        return TabConcat, TabSignal, TabReconstr, reconstruction
    
    def browse_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image_path.setText(file_name)
            self.original_image = cv2.imread(file_name)
            pixmap = QPixmap(file_name)
            self.image_display.setPixmap(pixmap.scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def recognize_digit(self):
        if self.original_image is None:
            self.prediction_label.setText("Please select an image first.")
            return

        # Create a copy of the original image for MNIST processing
        mnist_image = self.original_image.copy()

        # Preprocess image
        processed_image = self.preprocess_image(mnist_image)
        if processed_image is None:
            self.prediction_label.setText("No digit found in the image.")
            return

        print(f"Preprocessed image shape: {processed_image.shape}")  # Debug print

        # Make prediction
        image_tensor = transforms.ToTensor()(processed_image).unsqueeze(0).to(self.device)
        image_tensor = transforms.Normalize((0.5,), (0.5,))(image_tensor)

        with torch.no_grad():
            output = self.mnist_model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            predicted = torch.argmax(probabilities).item()

        # Display results
        self.prediction_label.setText(f"Predicted digit: {predicted}")
        self.plot_probability_distribution(probabilities.cpu().numpy())
        
    def process_image(self):
        if self.original_image is None:
            print("No image loaded.")

        # Create a copy of the original image for processing
        image_to_process = self.original_image.copy()

        process = self.process_combo.currentText()

        if process == "Grayscale":
            processed = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
        elif process == "Blur":
            processed = cv2.GaussianBlur(image_to_process, (15, 15), 0)
        elif process == "Edge Detection":
            gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
            processed = cv2.Canny(gray, 100, 200)

        # Convert processed image to QPixmap and display
        if len(processed.shape) == 2:  # Grayscale
            height, width = processed.shape
            qimage = QImage(processed.data, width, height, width, QImage.Format_Grayscale8)
        else:  # Color
            height, width, channel = processed.shape
            bytes_per_line = 3 * width
            qimage = QImage(processed.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimage)
        self.processed_display.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def preprocess_image(self, image):
        # Convert to grayscale if it's not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Invert if necessary (assuming dark digit on light background)
        if np.mean(gray) > 127:
            gray = 255 - gray
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find bounding box of largest contour
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            
            # Crop the image to the bounding box
            digit = thresh[y:y+h, x:x+w]
            
            # Add padding to make it square
            size = max(w, h)
            squared_digit = np.zeros((size, size), dtype=np.uint8)
            y_pos = (size - h) // 2
            x_pos = (size - w) // 2
            squared_digit[y_pos:y_pos+h, x_pos:x_pos+w] = digit
            
            # Resize to 28x28
            resized_digit = cv2.resize(squared_digit, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Ensure the digit is similar to MNIST (white digit on black background)
            if np.mean(resized_digit) > 127:
                resized_digit = 255 - resized_digit
            
            return resized_digit
        else:
            return None

    def plot_probability_distribution(self, probabilities):
        self.mnist_figure.clear()
        ax = self.mnist_figure.add_subplot(111)
        ax.bar(range(10), probabilities)
        ax.set_xlabel('Digit')
        ax.set_ylabel('Probability')
        ax.set_title('Probability Distribution')
        ax.set_xticks(range(10))
        self.mnist_canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FunctionGrapher()
    window.show()
    sys.exit(app.exec_())
