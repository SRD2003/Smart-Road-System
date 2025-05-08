import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import time
import random
import matplotlib.pyplot as plt

# Simulated Sensor Data Generation for Real-Time Simulation
def simulate_sensor_data():
    vibration_level = random.uniform(0, 10)  # Simulated vibration levels
    vehicle_count = random.randint(0, 50)   # Simulated vehicle count
    vehicle_speed = random.uniform(20, 120) # Simulated vehicle speed (km/h)
    road_stress = random.uniform(0, 100)    # Simulated road stress (MPa)
    road_strain = random.uniform(0, 0.05)   # Simulated road strain (unitless)
    pressure = random.uniform(50, 500)      # Simulated pressure on piezoelectric material (kPa)
    road_condition = random.choice(["Normal", "Pothole", "Damaged"])  # Simulated actual road condition
    return [vibration_level, vehicle_count, vehicle_speed, road_stress, road_strain, pressure, road_condition]

# Initialize Real-Time Data Arrays
sensor_data = []
power_generated = []
accuracies = []
latencies = []

# ML Model Initialization
# Simulate Training Data (Replace with actual training data)
data = {
    "vibration_level": np.random.uniform(0, 10, 1000),
    "vehicle_count": np.random.randint(0, 50, 1000),
    "vehicle_speed": np.random.uniform(20, 120, 1000),
    "road_stress": np.random.uniform(0, 100, 1000),
    "road_strain": np.random.uniform(0, 0.05, 1000),
    "pressure": np.random.uniform(50, 500, 1000),
    "road_condition": np.random.choice(["Normal", "Pothole", "Damaged"], 1000),
    "power_output": np.random.uniform(0, 50, 1000),
}

# Create DataFrame
df = pd.DataFrame(data)

# Train-Test Split for Classification (Road Condition)
X_class = df[["vibration_level", "vehicle_count", "vehicle_speed", "road_stress", "road_strain"]]
y_class = df["road_condition"]
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_class, y_train_class)

# Evaluate Initial Accuracy
y_pred_class = classifier.predict(X_test_class)
initial_accuracy = accuracy_score(y_test_class, y_pred_class)
accuracies.append(initial_accuracy)

# Train-Test Split for Regression (Power Output)
X_reg = df[["vehicle_speed", "pressure"]]
y_reg = df["power_output"]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_reg, y_train_reg)

# Evaluate Latency
def measure_latency():
    start_time = time.time()
    classifier.predict([simulate_sensor_data()[:5]])
    end_time = time.time()
    latency = end_time - start_time
    latencies.append(latency)

# Real-Time Simulation
def real_time_simulation():
    print("Starting Real-Time Simulation...")
    correct_predictions = 0
    total_predictions = 0

  for _ in range(20):  # Simulate 20 real-time iterations
        # Generate Sensor Data
        sensor_reading = simulate_sensor_data()
        sensor_data.append(sensor_reading[:6])

  # Predict Road Condition
  predicted_condition = classifier.predict([sensor_reading[:5]])[0]
        actual_condition = sensor_reading[6]

  # Update Accuracy
   if predicted_condition == actual_condition:
            correct_predictions += 1
        total_predictions += 1
        accuracies.append(correct_predictions / total_predictions)

  #Predict Power Generation
        power = regressor.predict([[sensor_reading[2], sensor_reading[5]]])[0]
        power_generated.append(power)

  # Measure Latency
  measure_latency()

  #Display Real-Time Results
        print(f"\nSensor Readings:")
        print(f"Vibration Level: {sensor_reading[0]:.2f}")
        print(f"Vehicle Count: {sensor_reading[1]}")
        print(f"Vehicle Speed: {sensor_reading[2]:.2f} km/h")
        print(f"Road Stress: {sensor_reading[3]:.2f} MPa")
        print(f"Road Strain: {sensor_reading[4]:.5f}")
        print(f"Pressure on Piezo: {sensor_reading[5]:.2f} kPa")
        print(f"Predicted Road Condition: {predicted_condition}")
        print(f"Actual Road Condition: {actual_condition}")
        print(f"Power Generated: {power:.2f} Wh")

  time.sleep(1)  # Simulate 1-second delay between iterations

# Plot Results
def plot_results():
    plt.figure(figsize=(12, 5))

#Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(len(accuracies)), accuracies, label='Accuracy', color='blue')
    plt.title('System Accuracy Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()

#Latency Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(len(latencies)), latencies, label='Latency (s)', color='red')
    plt.title('System Latency Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Latency (seconds)')
    plt.legend()

plt.tight_layout()
plt.show()

# Run Real-Time Simulation
real_time_simulation()

# Plot Accuracy and Latency
plot_results()
