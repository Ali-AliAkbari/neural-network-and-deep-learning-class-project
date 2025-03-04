# Variational Autoencoder (VAE) for Cartoon Face Generation

## Overview
This repository contains a PyTorch implementation of a Variational Autoencoder (VAE) trained on the Google Cartoon Set dataset. The model is designed to encode and decode cartoon face images, learning a lower-dimensional latent representation.

## Dataset
The dataset used is the **Google Cartoon Set**, available on Kaggle:
[Cartoon Faces - Google's Cartoon Set](https://www.kaggle.com/datasets)

Images are loaded from the dataset, preprocessed, and used for training the VAE model.

## Dependencies
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision numpy pandas matplotlib
```

## Code Structure
- `image_paths()`: Generates full paths for images in the dataset.
- `plot_sample()`: Displays a sample of images from the dataset.
- `VAE` class: Defines the Variational Autoencoder model.
- `CustomDataset` class: Custom dataset class for loading images.
- `train_model()`: Training function for the VAE.
- `decoder_img()`: Decodes a latent vector into an image.
- `plot()`: Plots training loss curves.

## Model Architecture
The VAE consists of:
- **Encoder**: A series of convolutional layers that reduce the image to a latent space representation.
- **Latent Space**: A compressed representation of the input image.
- **Decoder**: A series of transposed convolutional layers to reconstruct the image.

## Training
The model is trained using the **Mean Squared Error (MSE) loss** combined with a **Kullback-Leibler (KL) divergence loss**.

To train the model:
```python
optimizer = optim.Adam(model_cartoon.parameters(), lr=2e-6)
criterion = F.mse_loss
train_losses_epoch, train_losses_recon = train_model(model_cartoon, dataloader_cartoon, optimizer, criterion, device, num_epochs=5)
```

## Saving & Loading the Model
After training, the model can be saved and loaded using:
```python
torch.save(model_cartoon.state_dict(), 'model_scripted.pt')
```

## Generating New Images
To generate an image from a latent vector:
```python
z = torch.rand(35).to(device)*-1.9 + torch.exp(1*b[0]/2)
decoder_img(model_cartoon, z, device)
```
This will produce a new cartoon face from the learned latent distribution.

## Results
Training loss is plotted using:
```python
plot(num_epochs, train_losses_epoch, train_losses_recon)
```
<p align="center">
  <img src="Images/1.png"  width="400"/>
  <img src="Images/2.png"  width="400"/>
</p>

## Acknowledgments
- **Kaggle** for providing the dataset.
- **PyTorch** for deep learning frameworks.

## License
This project is licensed under the MIT License.

