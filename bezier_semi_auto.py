import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import feature

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian filter for noise reduction
    img_smooth = gaussian_filter(img, sigma=2)
    
    # Apply Canny edge detection with automatic threshold detection
    edges = feature.canny(img_smooth, sigma=3)
    
    return img, edges

def detect_surface(edges):
    height, width = edges.shape
    bottom_third = edges[2*height//3:]
    
    # Use Hough Line Transform to detect the surface line
    lines = cv2.HoughLines(bottom_third.astype(np.uint8), 1, np.pi/180, threshold=50)
    
    if lines is not None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            
            # Adjust y-coordinates to account for the bottom third crop
            y1 += 2*height//3
            y2 += 2*height//3
            
            return (x1, y1), (x2, y2)
    
    return None

def find_contact_points(edges, surface_line):
    height, width = edges.shape
    (x1, y1), (x2, y2) = surface_line
    
    # Calculate the equation of the surface line
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    else:
        m, b = None, x1  # Vertical line
    
    left_points = []
    right_points = []
    
    for x in range(width):
        if m is not None:
            y = int(m * x + b)
        else:
            y = int(b)
        
        if 0 <= y < height:
            if edges[y, x] > 0:
                if x < width // 2:
                    left_points.append((x, y))
                else:
                    right_points.append((x, y))
    
    left_point = min(left_points, key=lambda p: p[0]) if left_points else None
    right_point = max(right_points, key=lambda p: p[0]) if right_points else None
    
    return left_point, right_point

def calculate_contact_angle(edge_image, contact_point, is_left):
    if contact_point is None:
        return None
    
    x, y = contact_point
    height, width = edge_image.shape
    
    # Define search range and step
    search_range = 50
    step = 1
    angles = []
    
    for i in range(step, search_range, step):
        if is_left:
            x2, y2 = x + i, y
            while y2 > 0 and not edge_image[y2, x2]:
                y2 -= 1
        else:
            x2, y2 = x - i, y
            while y2 > 0 and not edge_image[y2, x2]:
                y2 -= 1
        
        if y2 > 0:
            dx = x2 - x
            dy = y - y2
            angle = np.degrees(np.arctan2(dy, dx))
            angles.append(angle)
    
    # Filter out outliers
    if angles:
        angles = np.array(angles)
        median = np.median(angles)
        mad = np.median(np.abs(angles - median))
        filtered_angles = angles[np.abs(angles - median) < 2 * mad]
        return np.mean(filtered_angles) if filtered_angles.size > 0 else None
    else:
        return None

def analyze_wettability(image_path):
    # Preprocess the image
    original, edges = preprocess_image(image_path)
    
    # Detect surface
    surface_line = detect_surface(edges)
    if surface_line is None:
        raise ValueError("Could not detect surface line")
    
    # Find contact points
    left_point, right_point = find_contact_points(edges, surface_line)
    
    # Calculate contact angles
    left_angle = calculate_contact_angle(edges, left_point, is_left=True)
    right_angle = calculate_contact_angle(edges, right_point, is_left=False)
    
    # Average contact angle (only if both angles are valid)
    if left_angle is not None and right_angle is not None:
        avg_angle = (left_angle + right_angle) / 2
    elif left_angle is not None:
        avg_angle = left_angle
    elif right_angle is not None:
        avg_angle = right_angle
    else:
        avg_angle = None
    
    return avg_angle, left_angle, right_angle, left_point, right_point, surface_line, original, edges

def visualize_results(original, edges, left_point, right_point, surface_line, left_angle, right_angle, avg_angle):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image with results
    ax1.imshow(original, cmap='gray')
    ax1.plot([surface_line[0][0], surface_line[1][0]], [surface_line[0][1], surface_line[1][1]], 'r-', linewidth=2)
    
    if left_point:
        ax1.plot(left_point[0], left_point[1], 'go', markersize=10)
    if right_point:
        ax1.plot(right_point[0], right_point[1], 'go', markersize=10)
    
    # Draw angle lines
    if left_angle is not None and left_point:
        left_angle_line = ((left_point[0], left_point[1]), 
                           (left_point[0] - 50, left_point[1] - int(50 * np.tan(np.radians(left_angle)))))
        ax1.plot([left_angle_line[0][0], left_angle_line[1][0]], [left_angle_line[0][1], left_angle_line[1][1]], 'y-', linewidth=2)
    
    if right_angle is not None and right_point:
        right_angle_line = ((right_point[0], right_point[1]), 
                            (right_point[0] + 50, right_point[1] - int(50 * np.tan(np.radians(right_angle)))))
        ax1.plot([right_angle_line[0][0], right_angle_line[1][0]], [right_angle_line[0][1], right_angle_line[1][1]], 'y-', linewidth=2)
    
    ax1.set_title('Original Image with Analysis')
    
    # Edge image
    ax2.imshow(edges, cmap='gray')
    ax2.set_title('Edge Detection')
    
    angle_text = f'Average Angle: {avg_angle:.2f}°\n' if avg_angle is not None else ''
    angle_text += f'Left Angle: {left_angle:.2f}°\n' if left_angle is not None else 'Left Angle: N/A\n'
    angle_text += f'Right Angle: {right_angle:.2f}°' if right_angle is not None else 'Right Angle: N/A'
    plt.suptitle(angle_text)
    plt.show()

# Example usage

image_path = r'C:\Users\yassine\Desktop\prj\capture.jpg'
avg_angle, left_angle, right_angle, left_point, right_point, surface_line, original, edges = analyze_wettability(image_path)
visualize_results(original, edges, left_point, right_point, surface_line, left_angle, right_angle, avg_angle)
print(f"Left Angle: {left_angle:.2f}° (if None, measurement failed)")
print(f"Right Angle: {right_angle:.2f}°")