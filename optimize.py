import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import matplotlib.animation as animation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = r"\data\processed_KDDTest.csv"
data = pd.read_csv(data_path)

# Define QoS priority levels using a dictionary
SERVICE_PRIORITY_MAPPING = {
    'http': 'High Priority', 'video_stream': 'High Priority', 'telnet': 'High Priority', 'domain_u': 'High Priority',
    'ftp': 'Medium Priority', 'smtp': 'Medium Priority', 'pop_3': 'Medium Priority', 'uucp': 'Medium Priority', 'eco_i': 'Medium Priority',
    'other': 'Low Priority', 'remote_job': 'Low Priority', 'gopher': 'Low Priority', 'whois': 'Low Priority', 'link': 'Low Priority',
    'X_11': 'Low Priority', 'discard': 'Low Priority', 'ctf': 'Low Priority', 'nss': 'Low Priority', 'daytime': 'Low Priority'
}

def qos_priority(service):
    """Assign QoS priority based on service type."""
    return SERVICE_PRIORITY_MAPPING.get(service, 'Low Priority')

# Add QoS prioritization
data['qos_priority'] = data['service'].apply(qos_priority)

# Bandwidth allocation rules
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

# Optimize traffic using Gradient Boosting Classifier
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
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, le

model, label_encoder = optimize_traffic(data)

# Visualizations
def plot_sankey(data):
    """Plot Sankey Diagram."""
    source = []
    target = []
    value = []
    for _, row in data.iterrows():
        source.append(row['qos_priority'])
        target.append(row['bandwidth'])
        value.append(row['dst_host_rerror_rate'] * 100)

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=list(data['qos_priority'].unique()) + list(data['bandwidth'].unique()),
            color="blue"
        ),
        link=dict(
            source=[list(data['qos_priority'].unique()).index(s) for s in source],
            target=[list(data['bandwidth'].unique()).index(t) + len(data['qos_priority'].unique()) for t in target],
            value=value
        )
    ))
    fig.update_layout(title_text="Traffic Flow Optimization - Sankey Diagram", font_size=10)
    fig.show()

def plot_heatmap(data):
    """Plot Heatmap of Error Rates by QoS and Bandwidth."""
    pivot_data = data.pivot_table(values='dst_host_rerror_rate', index='qos_priority', columns='bandwidth', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Heatmap of Error Rates by QoS Priority and Bandwidth Allocation")
    plt.xlabel("Bandwidth Allocation")
    plt.ylabel("QoS Priority")
    plt.show()

def plot_traffic_animation(data):
    """Animate Traffic Flow."""
    fig, ax = plt.subplots(figsize=(10, 6))
    traffic_flow = nx.DiGraph()

    for _, row in data.iterrows():
        traffic_flow.add_edge(row['qos_priority'], row['bandwidth'], weight=row['dst_host_rerror_rate'])

    pos = nx.spring_layout(traffic_flow)
    def update(num):
        ax.clear()
        nx.draw(traffic_flow, pos, with_labels=True, ax=ax, node_color="lightgreen", edge_color="gray", node_size=3000, font_size=10)
        nx.draw_networkx_edges(traffic_flow, pos, alpha=num / 10, ax=ax)

    ani = animation.FuncAnimation(fig, update, frames=10, repeat=True)
    plt.show()

def plot_temporal_line_graph(data):
    """Plot Temporal Error Rate Over Time."""
    time_data = data.groupby('service')['dst_host_rerror_rate'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(time_data['service'], time_data['dst_host_rerror_rate'], label='Error Rate Over Time', color='blue', marker='o')
    plt.title("Temporal Error Rate Before and After Optimization")
    plt.xlabel("Service")
    plt.ylabel("Average Error Rate")
    plt.legend()
    plt.grid()
    plt.show()

# Call visualizations
plot_sankey(data)
plot_heatmap(data)
plot_traffic_animation(data)
plot_temporal_line_graph(data)
