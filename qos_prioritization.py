# qos_prioritization.py
import pandas as pd

# Define QoS priority levels using a dictionary
SERVICE_PRIORITY_MAPPING = {
    # High Priority Services
    'http': 'High Priority',
    'video_stream': 'High Priority',
    'telnet': 'High Priority',
    'domain_u': 'High Priority',
    
    # Medium Priority Services
    'ftp': 'Medium Priority',
    'smtp': 'Medium Priority',
    'pop_3': 'Medium Priority',
    'uucp': 'Medium Priority',
    'eco_i': 'Medium Priority',
    
    # Low Priority Services
    'other': 'Low Priority',
    'remote_job': 'Low Priority',
    'gopher': 'Low Priority',
    'whois': 'Low Priority',
    'link': 'Low Priority',
    'X_11': 'Low Priority',
    'discard': 'Low Priority',
    'ctf': 'Low Priority',
    'nss': 'Low Priority',
    'daytime': 'Low Priority',  # Example of unspecified but common service
}

def qos_priority(service):
    """
    Assign QoS priority based on service type.
    Any service not explicitly listed will default to Low Priority.
    """
    return SERVICE_PRIORITY_MAPPING.get(service, 'Low Priority')

def apply_qos_prioritization(data):
    """Add QoS prioritization column and display distribution."""
    # Apply QoS prioritization to the dataset
    data['qos_priority'] = data['service'].apply(qos_priority)
    
    # Display QoS Priority Distribution
    priority_counts = data['qos_priority'].value_counts()
    print("QoS Priority Distribution:\n", priority_counts)

    # Display specific service and priority information
    print("\nServices with QoS Priority from data:")
    print(data[['service', 'qos_priority']].head(10))

# Load data
data_path = r"\data\processed_KDDTest.csv"
data = pd.read_csv(data_path)

# Apply QoS prioritization
apply_qos_prioritization(data)
