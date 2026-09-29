#import torch
#import torch.nn as nn
#import torch.optim as optim
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from torch.utils.data import DataLoader, TensorDataset
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import r2_score
#
## ==========================================
## 1. DYNAMIC MODEL DEFINITION (Standard MLP)
## ==========================================
#class DynamicCuGradeANN(nn.Module):
#    def __init__(self, input_dim, hidden_layers, dropout_rate=0.2, activation_name='ReLU'):
#        super(DynamicCuGradeANN, self).__init__()
#        
#        activations = {
#            'ReLU': nn.ReLU(),
#            'Tanh': nn.Tanh(),
#            'LeakyReLU': nn.LeakyReLU()
#        }
#        activation = activations.get(activation_name, nn.ReLU())
#        
#        layers = []
#        in_features = input_dim
#        
#        for out_features in hidden_layers:
#            if out_features > 0: 
#                layers.append(nn.Linear(in_features, out_features))
#                layers.append(activation)
#                layers.append(nn.Dropout(dropout_rate))
#                in_features = out_features
#                
#        layers.append(nn.Linear(in_features, 1))
#        self.network = nn.Sequential(*layers)
#
#    def forward(self, x):
#        return self.network(x)
#
## ==========================================
## 2. EVALUATION METRICS FUNCTION
## ==========================================
#def evaluate_comprehensive_metrics(model, data_loader, device):
#    model.eval()
#    mse_criterion = nn.MSELoss()
#    mae_criterion = nn.L1Loss()
#    huber_criterion = nn.HuberLoss()
#    
#    total_mse, total_mae, total_huber = 0.0, 0.0, 0.0
#    all_targets, all_predictions = [], []
#    
#    with torch.no_grad():
#        for batch_x, batch_y in data_loader:
#            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#            outputs = model(batch_x)
#            
#            total_mse += mse_criterion(outputs, batch_y).item() * batch_x.size(0)
#            total_mae += mae_criterion(outputs, batch_y).item() * batch_x.size(0)
#            total_huber += huber_criterion(outputs, batch_y).item() * batch_x.size(0)
#            
#            all_predictions.extend(outputs.cpu().numpy())
#            all_targets.extend(batch_y.cpu().numpy())
#            
#    num_samples = len(data_loader.dataset)
#    final_mse = total_mse / num_samples
#    final_rmse = np.sqrt(final_mse)
#    final_mae = total_mae / num_samples
#    final_huber = total_huber / num_samples
#    final_r2 = r2_score(all_targets, all_predictions)
#    
#    return {
#        'MSE': final_mse,
#        'RMSE': final_rmse,
#        'MAE': final_mae,
#        'Huber Loss': final_huber,
#        'R2 Score': final_r2
#    }
#
#
## ==========================================
## 3. MAIN EXECUTION BLOCK
## ==========================================
#if __name__ == "__main__":
#    # Define device globally
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    print(f"Using device: {device}")
#
#    # ---------------------------------------------------------
#    # [!] PLUGGED IN BEST HYPERPARAMETERS FROM BAYESIAN OPTIMIZATION [!]
#    # ---------------------------------------------------------
#    best_hidden_layers = [83, 203]                 # Based on 'n_layers': 2, 'n_units_l0': 83, 'n_units_l1': 203
#    best_lr = 0.0006897050285274453                
#    best_dropout = 0.04946013857483439             
#    best_batch_size = 32           
#    best_optimizer_name = 'Adam'   
#    best_activation = 'LeakyReLU'       
#    final_epochs = 150             
#    # ---------------------------------------------------------
#
#    # --- A. Load and Prepare Data ---
#    print("Loading and preparing data...")
#    merged = pd.read_excel(r'./merged_cleaned.xlsx') 
#    rock_cols = [col for col in merged.columns if col.startswith('Rock Class_')] 
#    
#    X_df = merged[
#        [
#            'Midpoint_Depth', 'Length', 'Ag_Grade', 'Azimuth', 'Dip', 
#            'Total depth', 'X', 'Y', 'Z', 'Depth_Ratio', 'Vertical_Depth'
#        ] + rock_cols
#    ] 
#    
#    # Target normalization via log transformation
#    y_series = np.log1p(merged['Cu_Grade']) 
#    
#    # Scale inputs to prevent exploding gradients
#    scaler_X = StandardScaler() 
#    X_scaled = scaler_X.fit_transform(X_df.values) 
#    
#    X_tensor = torch.tensor(X_scaled, dtype=torch.float32) 
#    y_tensor = torch.tensor(y_series.values, dtype=torch.float32).unsqueeze(1) 
#    
#    input_dim = X_tensor.shape[1] 
#    
#    # --- B. Create Train/Val Split and DataLoaders ---
#    dataset = TensorDataset(X_tensor, y_tensor) 
#    
#    train_size = int(0.8 * len(dataset))
#    val_size = len(dataset) - train_size
#    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#    
#    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
#    val_loader = DataLoader(val_dataset, batch_size=best_batch_size)
#    
#    # --- C. Instantiate the Final Model ---
#    print("\n--- Initializing Final Model with Best Architecture ---")
#    final_model = DynamicCuGradeANN(
#        input_dim=input_dim,
#        hidden_layers=best_hidden_layers,
#        dropout_rate=best_dropout,
#        activation_name=best_activation
#    ).to(device)
#    
#    # Set the optimizer dynamically
#    if best_optimizer_name == 'Adam':
#        final_optimizer = optim.Adam(final_model.parameters(), lr=best_lr)
#    elif best_optimizer_name == 'RMSprop':
#        final_optimizer = optim.RMSprop(final_model.parameters(), lr=best_lr)
#    else:
#        final_optimizer = optim.SGD(final_model.parameters(), lr=best_lr, momentum=0.9)
#        
#    final_criterion = nn.MSELoss() 
#    
#    # Initialize lists to track loss for plotting
#    train_losses = []
#    val_losses = []
#    
#    # --- D. Final Training Loop ---
#    print(f"Training for {final_epochs} epochs...")
#    
#    for epoch in range(final_epochs):
#        # 1. Training Phase
#        final_model.train()
#        epoch_train_loss = 0.0
#        
#        for batch_x, batch_y in train_loader:
#            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#            
#            final_optimizer.zero_grad()
#            outputs = final_model(batch_x)
#            loss = final_criterion(outputs, batch_y)
#            loss.backward()
#            final_optimizer.step()
#            
#            epoch_train_loss += loss.item() * batch_x.size(0)
#            
#        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
#        train_losses.append(avg_train_loss)
#        
#        # 2. Validation Phase
#        final_model.eval()
#        epoch_val_loss = 0.0
#        with torch.no_grad():
#            for batch_x, batch_y in val_loader:
#                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#                outputs = final_model(batch_x)
#                loss = final_criterion(outputs, batch_y)
#                epoch_val_loss += loss.item() * batch_x.size(0)
#                
#        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
#        val_losses.append(avg_val_loss)
#        
#        # Print progress periodically
#        if (epoch + 1) % 10 == 0 or epoch == 0:
#            print(f"Epoch {epoch+1}/{final_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
#            
#    # Save the trained model weights
#    torch.save(final_model.state_dict(), 'best_cu_grade_model.pth') 
#    print("\n-> Final model weights successfully saved to 'best_cu_grade_model.pth'")
#    
#    # --- E. Plotting the Learning Curve ---
#    plt.figure(figsize=(10, 6))
#    plt.plot(range(1, final_epochs + 1), train_losses, label='Training Loss', color='blue')
#    plt.plot(range(1, final_epochs + 1), val_losses, label='Validation Loss', color='orange')
#    plt.xlabel('Epochs')
#    plt.ylabel('Loss (MSE)')
#    plt.title('Training and Validation Loss Over Epochs')
#    plt.legend()
#    plt.grid(True)
#    
#    # Save plot to current directory
#    plt.savefig('loss_curve.png')
#    print("-> Loss curve successfully saved to 'loss_curve.png'")
#    
#    # --- F. Comprehensive Final Evaluation ---
#    print("\n=========================================")
#    print("FINAL MODEL PERFORMANCE METRICS (ON VAL SET)")
#    print("=========================================")
#    
#    # Evaluate the finalized model on the validation set
#    final_metrics = evaluate_comprehensive_metrics(final_model, val_loader, device)
#    
#    for metric_name, value in final_metrics.items():
#        print(f"{metric_name}: {value:.6f}")



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ==========================================
# 1. DYNAMIC MODEL DEFINITION (Standard MLP)
# ==========================================
class DynamicCuGradeANN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.2, activation_name='ReLU'):
        super(DynamicCuGradeANN, self).__init__()
        
        activations = {
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh(),
            'LeakyReLU': nn.LeakyReLU()
        }
        activation = activations.get(activation_name, nn.ReLU())
        
        layers = []
        in_features = input_dim
        
        for out_features in hidden_layers:
            if out_features > 0: 
                layers.append(nn.Linear(in_features, out_features))
                layers.append(activation)
                layers.append(nn.Dropout(dropout_rate))
                in_features = out_features
                
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ==========================================
# 2. EVALUATION METRICS FUNCTION
# ==========================================
def evaluate_comprehensive_metrics(model, data_loader, device):
    """
    Evaluates the model and returns MSE, RMSE, MAE, Huber Loss, and R2 Score.
    """
    model.eval()
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    huber_criterion = nn.HuberLoss()
    
    total_mse, total_mae, total_huber = 0.0, 0.0, 0.0
    all_targets, all_predictions = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            total_mse += mse_criterion(outputs, batch_y).item() * batch_x.size(0)
            total_mae += mae_criterion(outputs, batch_y).item() * batch_x.size(0)
            total_huber += huber_criterion(outputs, batch_y).item() * batch_x.size(0)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    num_samples = len(data_loader.dataset)
    final_mse = total_mse / num_samples
    final_rmse = np.sqrt(final_mse)
    final_mae = total_mae / num_samples
    final_huber = total_huber / num_samples
    final_r2 = r2_score(all_targets, all_predictions)
    
    return {
        'MSE': final_mse,
        'RMSE': final_rmse,
        'MAE': final_mae,
        'Huber Loss': final_huber,
        'R2 Score': final_r2
    }


# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Define device globally
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # [!] PLUGGED IN BEST HYPERPARAMETERS FROM BAYESIAN OPTIMIZATION [!]
    # ---------------------------------------------------------
    best_hidden_layers = [83, 203]                 
    best_lr = 0.0006897050285274453                
    best_dropout = 0.04946013857483439             
    best_batch_size = 32           
    best_optimizer_name = 'Adam'   
    best_activation = 'LeakyReLU'       
    final_epochs = 150             
    # ---------------------------------------------------------

    # --- A. Load and Prepare Data ---
    print("Loading and preparing data...")
    merged = pd.read_excel(r'./merged_cleaned.xlsx') 
    rock_cols = [col for col in merged.columns if col.startswith('Rock Class_')] 
    
    X_df = merged[
        [
            'Midpoint_Depth', 'Length', 'Ag_Grade', 'Azimuth', 'Dip', 
            'Total depth', 'X', 'Y', 'Z', 'Depth_Ratio', 'Vertical_Depth'
        ] + rock_cols
    ] 
    
    # Target normalization via log transformation
    y_series = np.log1p(merged['Cu_Grade']) 
    
    # Scale inputs to prevent exploding gradients
    scaler_X = StandardScaler() 
    X_scaled = scaler_X.fit_transform(X_df.values) 
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32) 
    y_tensor = torch.tensor(y_series.values, dtype=torch.float32).unsqueeze(1) 
    
    input_dim = X_tensor.shape[1] 
    
    # --- B. STRICT TRAIN / VAL SPLIT ---
    dataset = TensorDataset(X_tensor, y_tensor) 
    
    # Divide into 80% Train, 20% Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)
    
    # --- C. Instantiate the Final Model ---
    print("\n--- Initializing Final Model with Best Architecture ---")
    final_model = DynamicCuGradeANN(
        input_dim=input_dim,
        hidden_layers=best_hidden_layers,
        dropout_rate=best_dropout,
        activation_name=best_activation
    ).to(device)
    
    # Set the optimizer dynamically
    if best_optimizer_name == 'Adam':
        final_optimizer = optim.Adam(final_model.parameters(), lr=best_lr)
    elif best_optimizer_name == 'RMSprop':
        final_optimizer = optim.RMSprop(final_model.parameters(), lr=best_lr)
    else:
        final_optimizer = optim.SGD(final_model.parameters(), lr=best_lr, momentum=0.9)
        
    final_criterion = nn.MSELoss() 
    
    # Initialize lists to track loss for plotting
    train_losses = []
    val_losses = []
    
    # --- D. Final Training Loop ---
    print(f"Training for {final_epochs} epochs...")
    
    for epoch in range(final_epochs):
        # 1. Training Phase
        final_model.train()
        epoch_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            final_optimizer.zero_grad()
            outputs = final_model(batch_x)
            loss = final_criterion(outputs, batch_y)
            loss.backward()
            final_optimizer.step()
            
            epoch_train_loss += loss.item() * batch_x.size(0)
            
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # 2. Validation Phase (To track val_loss for the plot)
        final_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = final_model(batch_x)
                loss = final_criterion(outputs, batch_y)
                epoch_val_loss += loss.item() * batch_x.size(0)
                
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # Print progress periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{final_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
    # Save the trained model weights
    torch.save(final_model.state_dict(), 'best_cu_grade_model.pth') 
    print("\n-> Final model weights successfully saved to 'best_cu_grade_model.pth'")
    
    # --- E. Plotting the Learning Curve ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, final_epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, final_epochs + 1), val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Save plot to current directory
    plt.savefig('loss_curve.png')
    print("-> Loss curve successfully saved to 'loss_curve.png'")
    
    # --- F. Comprehensive Final Evaluation (ON VALIDATION SET ONLY) ---
    print("\n=========================================")
    print("FINAL MODEL PERFORMANCE METRICS (ON VAL SET)")
    print("=========================================")
    
    # Evaluate the finalized model STRICTLY on the unseen validation loader
    final_metrics = evaluate_comprehensive_metrics(final_model, val_loader, device)
    
    for metric_name, value in final_metrics.items():
        print(f"{metric_name}: {value:.6f}")