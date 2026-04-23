import torch
import torch.nn as nn
import numpy as np

class GestureCNN(nn.Module):
    def __init__(self, num_classes, grid_size=7):
        super(GestureCNN, self).__init__()
        
        # Simplified convolutional layers with fewer filters
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block (reduced filters)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # With 2 max pooling layers: 7×7 → 3×3 → 1×1
        conv_output_size = max(1, grid_size // (2**2))
        
        # Simplified fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * conv_output_size * conv_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def reshape_landmarks_for_cnn(landmarks, grid_size=7):
    """
    Reshape the 1D landmark array (63 values) into a 2D grid format suitable for CNN processing.
    """
    # Reshape the 63 values (21 landmarks x 3 coordinates) into 3 channels
    x_coords = landmarks[0::3]  # x coordinates
    y_coords = landmarks[1::3]  # y coordinates
    z_coords = landmarks[2::3]  # z coordinates
    
    # Create a spatial representation by placing landmarks in a grid
    x_channel = np.zeros((grid_size, grid_size))
    y_channel = np.zeros((grid_size, grid_size))
    z_channel = np.zeros((grid_size, grid_size))
    
    # Map the 21 landmarks to positions in the grid
    for i in range(21):
        # Scale coordinates to grid indices
        x_idx = min(int(x_coords[i] * (grid_size-1)), grid_size-1)
        y_idx = min(int(y_coords[i] * (grid_size-1)), grid_size-1)
        
        # Set the values in the grid
        x_channel[y_idx, x_idx] = x_coords[i]
        y_channel[y_idx, x_idx] = y_coords[i]
        z_channel[y_idx, x_idx] = z_coords[i]
    
    # Stack channels to create a 3-channel representation
    grid_representation = np.stack([x_channel, y_channel, z_channel], axis=0)
    return grid_representation

def predict_gesture(model, landmarks, device):
    """
    Make a prediction using the model.
    """
    # Reshape landmarks for CNN
    reshaped_landmarks = reshape_landmarks_for_cnn(landmarks)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(reshaped_landmarks, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()
