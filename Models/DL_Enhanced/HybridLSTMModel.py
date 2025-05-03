class HybridLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.0, time_step_out=1):
        super(HybridLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Shared feature extraction
        self.feature_layer = nn.Linear(hidden_size, hidden_size // 2)
        
        # Binary classification: rain or no rain
        self.rain_classifier = nn.Linear(hidden_size // 2, 1)  # Binary output (0=no rain, 1=rain)
        
        # Rain intensity classifier (conditionally used)
        self.intensity_classifier = nn.Linear(hidden_size // 2, 3)  # 3 classes: small, medium, heavy
        
        # Regression head for rain volume (conditionally used)
        self.regressor = nn.Linear(hidden_size // 2, time_step_out)
        
        self.time_step_out = time_step_out
        self.hidden_size = hidden_size

    def forward(self, x):
        # LSTM feature extraction
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # Get the last time step output
        
        # Extract shared features
        features = F.relu(self.feature_layer(last_out))
        
        # Binary classification: rain or no rain
        rain_logits = self.rain_classifier(features)
        rain_prob = torch.sigmoid(rain_logits)  # Probability of rain
        
        # Rain intensity classification (3 classes)
        intensity_logits = self.intensity_classifier(features)
        intensity_probs = F.softmax(intensity_logits, dim=1)  # Probabilities for each intensity class
        
        # Regression for rain amount
        regression_out = self.regressor(features)
        
        return {
            'rain_logits': rain_logits,
            'rain_prob': rain_prob,
            'intensity_logits': intensity_logits,
            'intensity_probs': intensity_probs,
            'regression': regression_out
        }

# Custom loss function that combines binary classification, multiclass classification, and regression
class HybridRainfallLoss(nn.Module):
    def __init__(self, rain_threshold=0.1, class_boundaries=[0.0, 2.5, 6.0], 
                 binary_weight=1.0, intensity_weight=1.0, regression_weight=1.0):
        super(HybridRainfallLoss, self).__init__()
        self.rain_threshold = rain_threshold
        self.class_boundaries = class_boundaries
        
        # Loss weights
        self.binary_weight = binary_weight
        self.intensity_weight = intensity_weight
        self.regression_weight = regression_weight
        
        # Individual loss functions
        self.binary_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.intensity_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.regression_loss_fn = nn.MSELoss(reduction='mean')
    
    def forward(self, outputs, targets):
        # Extract model outputs
        rain_logits = outputs['rain_logits']
        intensity_logits = outputs['intensity_logits']
        regression = outputs['regression']
        
        # Prepare target values
        # Binary rain targets (1 if rain > threshold, 0 otherwise)
        binary_targets = (targets > self.rain_threshold).float()
        
        # Intensity class targets (0=small, 1=medium, 2=heavy)
        # Create class masks
        small_mask = (targets > self.class_boundaries[0]) & (targets <= self.class_boundaries[1])
        medium_mask = (targets > self.class_boundaries[1]) & (targets <= self.class_boundaries[2])
        heavy_mask = (targets > self.class_boundaries[2])
        
        # Convert to class indices (0, 1, or 2)
        intensity_targets = torch.zeros_like(targets, dtype=torch.long)
        intensity_targets[medium_mask] = 1
        intensity_targets[heavy_mask] = 2
        
        # Only consider intensity loss for samples where rain exists
        valid_intensity_mask = binary_targets.bool().squeeze()
        
        # Calculate losses
        binary_loss = self.binary_loss_fn(rain_logits, binary_targets)
        
        # Only calculate intensity loss on rainy samples (if any exist)
        if valid_intensity_mask.sum() > 0:
            intensity_loss = self.intensity_loss_fn(
                intensity_logits[valid_intensity_mask], 
                intensity_targets[valid_intensity_mask].squeeze()
            )
        else:
            intensity_loss = torch.tensor(0.0, device=rain_logits.device)
        
        # For regression, we can either:
        # 1. Apply regression loss to all samples
        # 2. Apply regression loss only to rainy samples
        # Let's use option 2 for better focus on rain prediction
        if valid_intensity_mask.sum() > 0:
            regression_loss = self.regression_loss_fn(
                regression[valid_intensity_mask], 
                targets[valid_intensity_mask]
            )
        else:
            regression_loss = torch.tensor(0.0, device=rain_logits.device)
        
        # Combine losses with weights
        total_loss = (
            self.binary_weight * binary_loss +
            self.intensity_weight * intensity_loss +
            self.regression_weight * regression_loss
        )
        
        return {
            'total_loss': total_loss,
            'binary_loss': binary_loss.item(),
            'intensity_loss': intensity_loss.item() if isinstance(intensity_loss, torch.Tensor) else intensity_loss,
            'regression_loss': regression_loss.item() if isinstance(regression_loss, torch.Tensor) else regression_loss
        }

# Function to get final predictions combining classification and regression
def get_hybrid_predictions(outputs):
    rain_prob = outputs['rain_prob']
    intensity_probs = outputs['intensity_probs']
    regression = outputs['regression']
    
    # Binary decision: rain or no rain
    is_rain = rain_prob > 0.5
    
    # Get the most likely intensity class
    _, intensity_class = torch.max(intensity_probs, dim=1)
    
    # Initialize predictions with zeros (no rain)
    final_predictions = torch.zeros_like(regression)
    
    # Where rain is predicted, use regression values
    final_predictions[is_rain] = regression[is_rain]
    
    return {
        'is_rain': is_rain,
        'intensity_class': intensity_class,
        'regression': regression,
        'final_predictions': final_predictions
    }