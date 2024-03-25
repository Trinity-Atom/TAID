# Program Name: TAID Perception Block
# Author: Javier Bravo Acedo
# Date: 25/03/2024

# Description: The aim of this code is to execute the main flow of the program

import torch
from dataset import CustomDataset
from dataloader import create_data_loaders
from model import create_model, initialize_model
from train_model import train_model
from inference import inference
from visualize_results import visualize_results
 
def main():
    # Definir dispositivo (CPU o GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    # Definir transformaciones para el dataset
    transform = ...
 
    # Crear Dataset
    dataset = CustomDataset(data_dir='path/to/dataset', transform=transform)
 
    # Crear DataLoaders
    train_loader, val_loader = create_data_loaders(dataset, batch_size=32, num_workers=4)
 
    # Crear modelo y realizar inicialización personalizada si es necesario
    model = create_model(architecture='resnet50', pretrained=True)
    model = initialize_model(model, num_classes=10, activation_func=torch.nn.ReLU(), initialization={'weight': 0.1})
 
    # Definir criterio de pérdida y optimizador
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
    # Entrenar el modelo
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
 
    # Guardar el modelo entrenado
    torch.save(model.state_dict(), 'trained_model.pth')
 
    # Realizar inferencia en una imagen de muestra
    sample_image_path = 'path/to/sample/image.jpg'
    sample_image = Image.open(sample_image_path)
    sample_image_tensor = transform(sample_image).unsqueeze(0).to(device)
    outputs = inference(model, sample_image_tensor)
 
    # Visualizar los resultados de la inferencia
    visualize_results(sample_image, outputs)
 
if __name__ == "__main__":
    main()
