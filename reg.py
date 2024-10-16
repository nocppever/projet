import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

def generate_scatter_plot(n_points, slope, intercept, error_std):
    x = np.linspace(0, 10, n_points)
    y = slope * x + intercept + np.random.normal(0, error_std, n_points)
    return x, y

def calculate_entropy(y):
    hist, _ = np.histogram(y, bins='auto', density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def main():
    # Parameters
    n_points = 100
    true_slope = 2
    true_intercept = 1
    error_std = 0.5
    n_iterations = 50

    # Generate scatter plot and calculate regression
    x, y = generate_scatter_plot(n_points, true_slope, true_intercept, error_std)
    
    # Calculate regression using scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Plot scatter and regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.plot(x, slope * x + intercept, color='red', label=f'Regression (slope={slope:.2f})')
    plt.plot(x, true_slope * x + true_intercept, color='green', linestyle='--', label='True line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Random Scatter Plot with Regression Line')
    plt.legend()
    plt.show()

    # Calculate entropy and R-squared for multiple scatter plots with different slopes
    slopes = np.linspace(0, 5, n_iterations)
    entropies = []
    r_squared_values = []

    for s in slopes:
        x, y = generate_scatter_plot(n_points, s, true_intercept, error_std)
        entropies.append(calculate_entropy(y))
        
        # Calculate R-squared
        _, _, r_value, _, _ = stats.linregress(x, y)
        r_squared_values.append(r_value**2)

    # Plot entropy vs slope
    plt.figure(figsize=(10, 6))
    plt.plot(slopes, entropies)
    plt.xlabel('Slope')
    plt.ylabel('Entropy')
    plt.title('Entropy of Scatter Plots vs Regression Slope')
    plt.show()

    # Plot entropy vs R-squared
    plt.figure(figsize=(10, 6))
    plt.scatter(r_squared_values, entropies, alpha=0.5)
    plt.xlabel('Entropy')
    plt.ylabel('R-squared')
    plt.title('Entropy vs R-squared for Random Scatter Plots')
    plt.show()

if __name__ == "__main__":
    main()