import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from utils import *

def concat(layers):
    return torch.cat(layers, dim=1) 

class DecomNet(nn.Module):
    def __init__(self,layer_num,channel=64,kernel_size=3):
        super(DecomNet, self).__init__()
        self.layer_num = layer_num
        self.shallow_feature_extraction=nn.Conv2d(4,channel,kernel_size=kernel_size*3,padding=kernel_size//2)
        self.activated_layers = nn.ModuleList([nn.Conv2d(channel,channel,kernel_size=kernel_size,padding=kernel_size//2) for i in range(layer_num)])
        self.recon_layers=nn.Conv2d(channel,4,kernel_size=kernel_size,padding=kernel_size//2)
    def forward(self,input_im):
        input_max, _ = torch.max(input_im, dim=1, keepdim=True) #1*3*256*256
        input_im = concat([input_im, input_max]) #1*4*256*256
        conv=self.shallow_feature_extraction(input_im) #1*64*256*256
        for i in range(self.layer_num):
            conv = F.relu(self.activated_layers[i](conv))
        conv=self.recon_layers(conv)
        R=torch.sigmoid(conv[:,0:3,:,:])
        L=torch.sigmoid(conv[:,3:4,:,:])
        return R,L

class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        # Convolutional layers for down-sampling (encoding)
        self.conv0 = nn.Conv2d(4, channel, kernel_size, padding=kernel_size // 2)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=kernel_size // 2)
        # Deconvolutional layers for up-sampling (decoding)
        self.deconv1 = nn.Conv2d(channel, channel, kernel_size, padding=kernel_size // 2)
        self.deconv2 = nn.Conv2d(channel, channel, kernel_size, padding=kernel_size // 2)
        self.deconv3 = nn.Conv2d(channel, channel, kernel_size, padding=kernel_size // 2)
        # Fusion layer to combine features from different levels
        self.feature_fusion = nn.Conv2d(channel * 3, channel, 1, padding=0)
        # Output layer to generate the final enhanced illumination map
        self.output_layer = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, input_L, input_R):
        # Concatenate reflectance map and illumination map
        input_im = concat([input_R, input_L])
        # Encoding path: apply down-sampling convolutions
        conv0 = self.conv0(input_im)
        conv1 = F.relu(self.conv1(conv0))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # Decoding path: up-sample and combine with previous layers
        up1 = F.interpolate(conv3, size=(conv2.shape[2], conv2.shape[3]), mode='nearest')
        deconv1 = F.relu(self.deconv1(up1) + conv2)
        up2 = F.interpolate(deconv1, size=(conv1.shape[2], conv1.shape[3]), mode='nearest')
        deconv2 = F.relu(self.deconv2(up2) + conv1)
        up3 = F.interpolate(deconv2, size=(conv0.shape[2], conv0.shape[3]), mode='nearest')
        deconv3 = F.relu(self.deconv3(up3) + conv0)
        
        # Resize feature maps to match the output size and concatenate
        deconv1_resize = F.interpolate(deconv1, size=(deconv3.shape[2], deconv3.shape[3]), mode='nearest')
        deconv2_resize = F.interpolate(deconv2, size=(deconv3.shape[2], deconv3.shape[3]), mode='nearest')
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
        # Fuse features from different levels
        feature_fusion = self.feature_fusion(feature_gather)
        # Generate the enhanced illumination map
        output = self.output_layer(feature_fusion)
        return output

class LowlightEnhance(nn.Module):
    def __init__(self):
        super(LowlightEnhance, self).__init__()
        # Number of layers for DecomNet
        self.DecomNet_layer_num = 5
        # Initialize DecomNet and RelightNet
        self.DecomNet = DecomNet(layer_num=self.DecomNet_layer_num)
        self.RelightNet = RelightNet()

    def forward(self, input_low, input_high):
        # Decomposition
        R_low, I_low = self.DecomNet(input_low)
        R_high, I_high = self.DecomNet(input_high)

        # Relight
        I_delta = self.RelightNet(I_low, R_low)

        # Concatenate channels
        I_low_3 = torch.cat([I_low, I_low, I_low], dim=1)
        I_high_3 = torch.cat([I_high, I_high, I_high], dim=1)
        I_delta_3 = torch.cat([I_delta, I_delta, I_delta], dim=1)

        output_R_low = R_low
        output_I_low = I_low_3
        output_I_delta = I_delta_3
        output_S = R_low * I_delta_3

        return output_R_low, output_I_low, output_I_delta, output_S, R_high, I_high_3, I_low, I_high, I_delta

    def loss(self, input_low, input_high, output_R_low, output_I_low, output_I_delta, output_S, R_high, I_high_3,I_low, I_high, I_delta):
        # Loss calculations
        recon_loss_low = torch.mean(torch.abs(output_R_low * output_I_low - input_low))
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - input_high))
        recon_loss_mutal_low = torch.mean(torch.abs(R_high * output_I_low - input_low))
        recon_loss_mutal_high = torch.mean(torch.abs(output_R_low * I_high_3 - input_high))
        equal_R_loss = torch.mean(torch.abs(output_R_low - R_high))
        relight_loss = torch.mean(torch.abs(output_R_low * output_I_delta - input_high))

        # Smoothness loss
        Ismooth_loss_low = self.smooth(I_low, output_R_low)
        Ismooth_loss_high = self.smooth(I_high, R_high)
        Ismooth_loss_delta = self.smooth(I_delta, output_R_low)

        # Total losses
        loss_Decom = recon_loss_low + recon_loss_high + 0.001 * recon_loss_mutal_low + \
                     0.001 * recon_loss_mutal_high + 0.1 * Ismooth_loss_low + 0.1 * Ismooth_loss_high + \
                     0.01 * equal_R_loss

        loss_Relight = relight_loss + 3 * Ismooth_loss_delta

        return loss_Decom, loss_Relight

    def gradient(self, input_tensor, direction):
        smooth_kernel_x = torch.Tensor([[0, 0], [-1, 1]]).unsqueeze(0).unsqueeze(0).to(input_tensor.device)
        smooth_kernel_y = smooth_kernel_x.transpose(1, 2)

        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y

        return torch.abs(F.conv2d(input_tensor, kernel, padding=1))

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R_gray = torch.mean(input_R, dim=1, keepdim=True)  # Convert to grayscale
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R_gray, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R_gray, "y")))

    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print(f"[*] Evaluating for phase {train_phase} / epoch {epoch_num}...")

        for idx, input_low_eval in enumerate(eval_low_data):
            input_low_eval = torch.unsqueeze(input_low_eval, 0)

            if train_phase == "Decom":
                result_1, result_2 = self(input_low_eval, input_low_eval)[:2]
            if train_phase == "Relight":
                result_1, result_2 = self(input_low_eval, input_low_eval)[2:]

            save_images(os.path.join(sample_dir, f'eval_{train_phase}_{idx + 1}_{epoch_num}.png'), result_1, result_2)

    def train_model(self, train_low_data, train_high_data, eval_low_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)

        print(f"[*] Start training for phase {train_phase}.")

        start_time = time.time()
        image_id = 0

        for epoch in range(epoch):
            for batch_id in range(len(train_low_data) // batch_size):
                # Generate data for a batch
                batch_input_low = torch.zeros(batch_size, 3, patch_size, patch_size)
                batch_input_high = torch.zeros(batch_size, 3, patch_size, patch_size)

                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)

                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x:x + patch_size, y:y + patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x:x + patch_size, y:y + patch_size, :], rand_mode)

                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data = zip(*tmp)

                optimizer.zero_grad()

                output_R_low, output_I_low, output_I_delta, output_S, R_high, I_high_3, I_low, I_high, I_delta = self(batch_input_low, batch_input_high)
                loss_Decom, loss_Relight = self.loss(batch_input_low, batch_input_high, output_R_low, output_I_low, output_I_delta, output_S, R_high, I_high_3,I_low, I_high, I_delta)

                # Choose which loss to backpropagate depending on the phase
                if train_phase == "Decom":
                    loss_Decom.backward()
                    optimizer.step()
                elif train_phase == "Relight":
                    loss_Relight.backward()
                    optimizer.step()

                print(f"{train_phase} Epoch: [{epoch + 1}] [{batch_id + 1}/{len(train_low_data) // batch_size}] time: {time.time() - start_time:.4f}, loss: {loss_Decom.item():.6f}")

            # Evaluate and save a checkpoint file for the model
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, sample_dir, train_phase)
                self.save_checkpoint(ckpt_dir, f"RetinexNet-{train_phase}", epoch)

    def save_checkpoint(self, ckpt_dir, model_name, epoch):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        checkpoint_path = os.path.join(ckpt_dir, f"{model_name}_epoch_{epoch}.pth")
        torch.save(self.state_dict(), checkpoint_path)
        print(f"[*] Saving model {model_name} at {checkpoint_path}")
        
    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag, device='cuda'):
    # Set model to evaluation mode and move it to the appropriate device
        self.to(device)
        self.eval()

        # Ensure save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("[*] Testing...")

        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])

            # Extract file name and extension
            _, name = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            # Prepare input data (assuming test_low_data is a numpy array, convert it to PyTorch tensor)
            input_low_test = torch.from_numpy(np.expand_dims(test_low_data[idx], axis=0)).float().to(device)

            # Forward pass through the model (no gradient calculation required during testing)
            with torch.no_grad():
                R_low, I_low, I_delta, S = self(input_low_test, input_low_test)

            # Convert outputs back to CPU numpy arrays for saving
            R_low = R_low.cpu().numpy().squeeze(0)
            I_low = I_low.cpu().numpy().squeeze(0)
            I_delta = I_delta.cpu().numpy().squeeze(0)
            S = S.cpu().numpy().squeeze(0)

            # Save the outputs
            if decom_flag == 1:
                save_images(os.path.join(save_dir, f"{name}_R_low.{suffix}"), R_low)
                save_images(os.path.join(save_dir, f"{name}_I_low.{suffix}"), I_low)
                save_images(os.path.join(save_dir, f"{name}_I_delta.{suffix}"), I_delta)
            
            save_images(os.path.join(save_dir, f"{name}_S.{suffix}"), S)
        
        print(f"[*] Testing complete. Images saved in {save_dir}")



