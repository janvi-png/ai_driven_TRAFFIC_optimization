import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = r"\data\processed_KDDTest.csv"
data = pd.read_csv(data_path)

# Console Output: Initial Data Summary
print("Data Loaded Successfully.")
print(f"Shape of Dataset: {data.shape}")
print("Sample Data:\n", data.head())

# QoS Priority Mapping
SERVICE_PRIORITY_MAPPING = {
    'http': 'High Priority', 'video_stream': 'High Priority', 'telnet': 'High Priority', 'domain_u': 'High Priority',
    'ftp': 'Medium Priority', 'smtp': 'Medium Priority', 'pop_3': 'Medium Priority', 'uucp': 'Medium Priority', 'eco_i': 'Medium Priority',
    'other': 'Low Priority', 'remote_job': 'Low Priority', 'gopher': 'Low Priority', 'whois': 'Low Priority', 'link': 'Low Priority',
    'X_11': 'Low Priority', 'discard': 'Low Priority', 'ctf': 'Low Priority', 'nss': 'Low Priority', 'daytime': 'Low Priority'
}

def qos_priority(service):
    """Assign QoS priority based on service type."""
    return SERVICE_PRIORITY_MAPPING.get(service, 'Low Priority')

# Add QoS Prioritization
data['qos_priority'] = data['service'].apply(qos_priority)

# Console Output: QoS Assignment
print("\nQoS Prioritization Added:")
print(data[['service', 'qos_priority']].head())

# Bandwidth Allocation Logic
def allocate_bandwidth(row):
    qos = row['qos_priority']
    if qos == 'High Priority' and (row['src_bytes'] > 1 or row['dst_bytes'] > 1):
        return 'Increased Bandwidth'
    elif qos == 'Medium Priority' and (row['src_bytes'] == 1 or row['dst_bytes'] == 1):
        return 'Moderate Bandwidth'
    elif qos == 'Low Priority' or (row['src_bytes'] == 0 and row['dst_bytes'] == 0):
        return 'Reduced Bandwidth'
    else:
        return 'Normal Bandwidth'

data['bandwidth'] = data.apply(allocate_bandwidth, axis=1)

# Console Output: Bandwidth Allocation
print("\nBandwidth Allocation Completed:")
print(data[['qos_priority', 'src_bytes', 'dst_bytes', 'bandwidth']].head())

# Traffic Optimization Using Gradient Boosting Classifier
def optimize_traffic(data):
    features = ['src_bytes', 'dst_bytes', 'count', 'srv_count', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
    target = 'bandwidth'

    le = LabelEncoder()
    data[target] = le.fit_transform(data[target])

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("\nTraffic Optimization Results:")
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, le

model, label_encoder = optimize_traffic(data)

# Traffic Rerouting Visualization
def reroute_traffic(data):
    print("\nInitiating Traffic Rerouting Visualization...")
    rerouted_data = data.copy()
    rerouted_data['reroute_action'] = rerouted_data['dst_host_rerror_rate'].apply(
        lambda x: 'Rerouted' if x > 0.45 else 'No Change')

    traffic_graph = nx.DiGraph()
    for _, row in rerouted_data.iterrows():
        traffic_graph.add_edge(row['qos_priority'], row['reroute_action'], weight=row['dst_host_rerror_rate'])

    pos = nx.spring_layout(traffic_graph)
    plt.figure(figsize=(12, 8))
    nx.draw(traffic_graph, pos, with_labels=True, node_color='lightblue', font_size=10, node_size=3000, edge_color='gray')
    edge_labels = nx.get_edge_attributes(traffic_graph, 'weight')
    nx.draw_networkx_edge_labels(traffic_graph, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
    plt.title("Traffic Rerouting Based on Error Rate")
    plt.show()

# Visualize Temporal Trends
def plot_temporal_trends(data):
    print("\nGenerating Temporal Error Rate Trends...")
    time_data = data.groupby('service')['dst_host_rerror_rate'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(time_data['service'], time_data['dst_host_rerror_rate'], label='Error Rate Over Time', color='blue', marker='o')
    plt.title("Temporal Error Rate Trends by Service")
    plt.xlabel("Service")
    plt.ylabel("Average Error Rate")
    plt.legend()
    plt.grid()
    plt.show()

# Execute Rerouting Visualization and Trends
reroute_traffic(data)
plot_temporal_trends(data)

# Console Output: Final Rerouting Summary
rerouted_summary = data.groupby('reroute_action')['dst_host_rerror_rate'].mean()
print("\nFinal Rerouting Summary:")
print(rerouted_summary)
