import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch
import os



class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image into a 1D vector
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def inner_loop(model, loss_fn, X_train, y_train, learning_rate):
    # Forward pass: Calculate the loss for the task's training data
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    
    # Compute gradients for task-specific adaptation
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    
    # Update model parameters manually using gradient descent
    updated_params = [param - learning_rate * grad for param, grad in zip(model.parameters(), grads)]
    
    return updated_params

def outer_loop(meta_model, tasks, learning_rate_inner, learning_rate_outer, loss_fn, optimizer):
    meta_loss = 0.0

    # Loop through each task in the batch of tasks
    for task in tasks:
        X_train, y_train, X_test, y_test = task
        
        # Inner loop - Adapt the model to the specific task
        updated_params = inner_loop(meta_model, loss_fn, X_train, y_train, learning_rate_inner)
        
        # Create a new model with the updated parameters
        adapted_model = SimpleModel()
        with torch.no_grad():
            for param, updated_param in zip(adapted_model.parameters(), updated_params):
                param.copy_(updated_param)
        
        # Forward pass with the adapted model on the test set
        y_test_pred = adapted_model(X_test)
        task_loss = loss_fn(y_test_pred, y_test)
        meta_loss += task_loss
    
    # Meta-update - Update the meta-model parameters
    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()
    
    return meta_loss.item()

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Download and prepare dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

def create_task(train_dataset, num_samples=5):
    # Randomly sample images for the task (few-shot learning)
    indices = random.sample(range(len(train_dataset)), num_samples * 2)  # Samples for train/test
    train_indices, test_indices = indices[:num_samples], indices[num_samples:]
    
    train_data = Subset(train_dataset, train_indices)
    test_data = Subset(train_dataset, test_indices)
    
    X_train, y_train = zip(*[(data[0], data[1]) for data in train_data])
    X_test, y_test = zip(*[(data[0], data[1]) for data in test_data])
    
    X_train, y_train = torch.stack(X_train), torch.tensor(y_train)
    X_test, y_test = torch.stack(X_test), torch.tensor(y_test)
    
    return X_train, y_train, X_test, y_test

def maml_training(meta_model, train_dataset, num_tasks=10, num_epochs=10, learning_rate_inner=0.01, learning_rate_outer=0.001):
    optimizer = optim.Adam(meta_model.parameters(), lr=learning_rate_outer)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        tasks = []
        
        # Create multiple tasks
        for _ in range(num_tasks):
            task = create_task(train_dataset)
            tasks.append(task)
        
        # Run outer loop for meta-learning
        meta_loss = outer_loop(meta_model, tasks, learning_rate_inner, learning_rate_outer, loss_fn, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Meta-Loss: {meta_loss:.4f}")

# Load dataset
train_dataset, _ = load_data()

# Initialize meta model
meta_model = SimpleModel()

# Start training MAML
maml_training(meta_model, train_dataset, num_tasks=5, num_epochs=10, learning_rate_inner=0.01, learning_rate_outer=0.001)

