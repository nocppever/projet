import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import feature
from scipy.interpolate import splprep, splev

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_smooth = gaussian_filter(img, sigma=2)
    return img, img_smooth

def detect_edges(img_smooth):
    edges = feature.canny(img_smooth, sigma=3)
    return edges

def fit_spline(edges):
    # Get coordinates of edge pixels
    y, x = np.nonzero(edges)
    
    # Sort points to follow the contour
    points = np.column_stack((x, y))
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:,1] - center[1], points[:,0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    # Fit a spline to the sorted points
    tck, u = splprep([sorted_points[:,0], sorted_points[:,1]], s=0, per=1)
    
    # Generate points on the fitted spline
    spline_points = splev(np.linspace(0, 1, 1000), tck)
    
    return spline_points

def visualize_results(original, edges, spline_points):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    
    # Original image
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Edge detection result
    ax2.imshow(edges, cmap='gray')
    ax2.set_title('Edge Detection Result')
    ax2.axis('off')
    
    # Spline fit result
    ax3.imshow(original, cmap='gray')
    ax3.plot(spline_points[0], spline_points[1], 'r-', linewidth=2)
    ax3.set_title('Spline Fit Result')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

def main_analysis(image_path):
    # Preprocess the image
    original, img_smooth = preprocess_image(image_path)
    
    # Detect edges
    edges = detect_edges(img_smooth)
    
    # Fit spline to edges
    spline_points = fit_spline(edges)
    
    # Visualize results
    visualize_results(original, edges, spline_points)
    
    return edges, spline_points

# Example usage
image_path = r"C:\Users\Yassine\Downloads\Capture.JPG"
edge_result, spline_result = main_analysis(image_path)