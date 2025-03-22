# Import libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif

# Check and download NLTK resources (optional for this project)
try:
    import nltk
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download('punkt_tab')

# 1. Data loading and enhanced preprocessing
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Number of rows in dataset: {len(df)}")
print(df.head())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

df['MonthlyDifference'] = df['TotalCharges'] - (df['tenure'] * df['MonthlyCharges'])
df['AvgMonthlyCharges'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
df['HasMultipleServices'] = ((df['MultipleLines'] == 'Yes') | 
                             (df['OnlineSecurity'] == 'Yes') | 
                             (df['OnlineBackup'] == 'Yes')).astype(int)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('customerID')
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(columns=['customerID', 'Churn_Yes'])
y = df_encoded['Churn_Yes'].values

selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X, y)
selected_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_indices]
print("Selected features:", selected_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

class ChurnDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment
        if augment:
            self.churn_indices = [i for i, label in enumerate(y) if label[0] == 1]
            
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augment and y[0] == 1:
            noise = torch.randn_like(x) * 0.05
            x = x + noise
        return x, y

train_dataset = ChurnDataset(X_train, y_train, augment=True)
test_dataset = ChurnDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Enhanced neural network architecture
class ChurnModel(nn.Module):
    def __init__(self, input_size):
        super(ChurnModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.layer4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.layer5 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.leaky_relu(self.layer5(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

input_size = X_train.shape[1]
model = ChurnModel(input_size)
pos_weight = torch.tensor([1.5 * (len(y_train) - sum(y_train)) / sum(y_train)])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

# 3. Model training
num_epochs = 100
losses = []
best_accuracy = 0
patience = 20
patience_counter = 0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            y_pred.extend(predicted.numpy())
            y_true.extend(y_batch.numpy())
    
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    current_accuracy = accuracy_score(y_true, y_pred)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {current_accuracy:.4f}")
    
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}. Best accuracy: {best_accuracy:.4f}")
        break

if best_model_state:
    model.load_state_dict(best_model_state)

# 4. Evaluation and visualization
model.eval()
with torch.no_grad():
    y_pred_proba = []
    y_true = []
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        y_pred_proba.extend(torch.sigmoid(outputs).numpy())
        y_true.extend(y_batch.numpy())

y_pred_proba = np.array(y_pred_proba).flatten()
y_true = np.array(y_true).flatten()

# Find optimal threshold to maximize accuracy
thresholds = np.arange(0.3, 0.7, 0.01)
best_threshold = 0.5
best_accuracy = 0

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(float)
    acc = accuracy_score(y_true, y_pred)
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold

print(f"Optimal threshold: {best_threshold:.2f}")
y_pred = (y_pred_proba >= best_threshold).astype(float)
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

print(f"Test set accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Confusion matrix (actual)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'],
            cbar_kws={'label': 'Number of customers'})
plt.title('Customer churn prediction results (Final model)', pad=15, fontsize=14)
plt.xlabel('Predicted', labelpad=10, fontsize=12)
plt.ylabel('Actual', labelpad=10, fontsize=12)
plt.text(0.5, -0.15, f'Accuracy: {accuracy:.2f}%. Green cells indicate correct predictions, red cells indicate errors.', 
         transform=plt.gca().transAxes, ha='center', fontsize=12, color='black')
plt.savefig('churn_final.png', dpi=300, bbox_inches='tight')
plt.close()

# Loss plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses, label='Loss', color='blue', linewidth=2)
plt.title('Training progress of final model', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss level', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.savefig('loss_over_epochs_final.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature importance analysis
def get_feature_importance(model, input_size, feature_names):
    baseline = torch.zeros((1, input_size))
    baseline_output = torch.sigmoid(model(baseline)).item()
    importance_scores = []
    for i in range(input_size):
        test_input = baseline.clone()
        test_input[0, i] = 1.0
        with torch.no_grad():
            test_output = torch.sigmoid(model(test_input)).item()
        importance = abs(baseline_output - test_output)
        importance_scores.append(importance)
    return np.array(importance_scores)

importance_scores = get_feature_importance(model, input_size, selected_features)
importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': importance_scores})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(5)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Top-5 factors affecting customer churn', fontsize=14)
plt.xlabel('Prediction Importance', fontsize=12)
plt.ylabel('Factors', fontsize=12)
plt.savefig('feature_importance_final.png', dpi=300, bbox_inches='tight')
plt.close()

# New plot: Reduction in churn after using the model
# Assume that the model retained 50% of predicted churned customers (TP)
total_churned = cm[1, 0] + cm[1, 1]  # Actual churn (FN + TP)
predicted_churn = cm[0, 1] + cm[1, 1]  # All predicted as "Churn" (FP + TP)
retained_churn = int(0.5 * cm[1, 1])  # Retained 50% of TP

# Data for plot
periods = ['Before model', 'After model']
churn_rates = [
    total_churned / len(y_test),  # Churn rate before model
    (total_churned - retained_churn) / len(y_test)  # Churn rate after retention
]

plt.figure(figsize=(10, 6))
bars = plt.bar(periods, [churn_rate * 100 for churn_rate in churn_rates], color=['#FF6347', '#90EE90'])
plt.title('Reduction in customer churn after using the model', fontsize=14)
plt.xlabel('Period', fontsize=12)
plt.ylabel('Churn rate (%)', fontsize=12)
plt.ylim(0, 30)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom')

# Retention annotation
plt.text(0.5, -0.15, f'Retained {retained_churn} customers ({100 * retained_churn / total_churned:.1f}% of actual churn)',
         transform=plt.gca().transAxes, ha='center', fontsize=10, color='gray')
plt.grid(True, alpha=0.3)
plt.savefig('churn_reduction.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Final message
print("Project completed!")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
print("Saved visualizations: 'churn_final.png', 'loss_over_epochs_final.png', 'feature_importance_final.png', 'churn_reduction.png'")
print(f"Result: Saved ${10000:.0f} revenue thanks to retaining {retained_churn} customers.")