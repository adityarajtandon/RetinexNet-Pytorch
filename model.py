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
    # Print the shape of each tensor in the lis
    # Concatenate the tensors along the channel dimension (dim=1)
    a = torch.cat(layers, dim=1)

    # Print the shape of the concatenated result

    return a



class DecomNet(nn.Module):
    def __init__(self, layer_num, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.layer_num = layer_num
        self.shallow_feature_extraction = nn.Conv2d(4, channel, kernel_size=kernel_size*3, padding=(kernel_size*3)//2)
        self.activated_layers = nn.ModuleList([nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=kernel_size//2) for i in range(layer_num)])
        self.recon_layers = nn.Conv2d(channel, 4, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, input_im):
        print(f"{input_im.shape}")
        input_max, _ = torch.max(input_im, dim=1, keepdim=True)
        # print(f"Input max shape: {input_max.shape}")
        input_im = concat([input_im, input_max])
       #  print(f"Concatenated input shape: {input_im.shape}")

        conv = self.shallow_feature_extraction(input_im)
       #  print(f"Shallow feature extraction shape: {conv.shape}")
        for i in range(self.layer_num):
            conv = F.relu(self.activated_layers[i](conv))
           #  print(f"Conv layer {i+1} shape: {conv.shape}")

        conv = self.recon_layers(conv)
      #   print(f"Reconstruction layer shape: {conv.shape}")

        R = torch.sigmoid(conv[:, 0:3, :, :])
        L = torch.sigmoid(conv[:, 3:4, :, :])
       #  print(f"Reflectance (R) shape: {R.shape}, Illumination (L) shape: {L.shape}")
        return R, L


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        self.conv0 = nn.Conv2d(4, channel, kernel_size, padding=kernel_size // 2)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=kernel_size // 2)
        self.deconv1 = nn.Conv2d(channel, channel, kernel_size, padding=kernel_size // 2)
        self.deconv2 = nn.Conv2d(channel, channel, kernel_size, padding=kernel_size // 2)
        self.deconv3 = nn.Conv2d(channel, channel, kernel_size, padding=kernel_size // 2)
        self.feature_fusion = nn.Conv2d(channel * 3, channel, 1, padding=0)
        self.output_layer = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, input_L, input_R):
        input_im = concat([input_R, input_L])
       #  print(f"RelightNet input shape: {input_im.shape}")

        conv0 = self.conv0(input_im)
       #  print(f"Conv0 shape: {conv0.shape}")
        conv1 = F.relu(self.conv1(conv0))
        # print(f"Conv1 shape: {conv1.shape}")
        conv2 = F.relu(self.conv2(conv1))
        # print(f"Conv2 shape: {conv2.shape}")
        conv3 = F.relu(self.conv3(conv2))
        #  print(f"Conv3 shape: {conv3.shape}")

        up1 = F.interpolate(conv3, size=(conv2.shape[2], conv2.shape[3]), mode='nearest')
        # print(f"Up-sampled conv3 shape: {up1.shape}")
        deconv1 = F.relu(self.deconv1(up1) + conv2)
        # print(f"Deconv1 shape: {deconv1.shape}")
        up2 = F.interpolate(deconv1, size=(conv1.shape[2], conv1.shape[3]), mode='nearest')
       #  print(f"Up-sampled deconv1 shape: {up2.shape}")
        deconv2 = F.relu(self.deconv2(up2) + conv1)
        # print(f"Deconv2 shape: {deconv2.shape}")
        up3 = F.interpolate(deconv2, size=(conv0.shape[2], conv0.shape[3]), mode='nearest')
        # print(f"Up-sampled deconv2 shape: {up3.shape}")
        deconv3 = F.relu(self.deconv3(up3) + conv0)
       #  print(f"Deconv3 shape: {deconv3.shape}")

        deconv1_resize = F.interpolate(deconv1, size=(deconv3.shape[2], deconv3.shape[3]), mode='nearest')
        deconv2_resize = F.interpolate(deconv2, size=(deconv3.shape[2], deconv3.shape[3]), mode='nearest')
        # print(f"Deconv1 resize shape: {deconv1_resize.shape}")
       #  print(f"Deconv2 resize shape: {deconv2_resize.shape}")

        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
       #  print(f"Feature gather shape: {feature_gather.shape}")
        feature_fusion = self.feature_fusion(feature_gather)
       #  print(f"Feature fusion shape: {feature_fusion.shape}")
        output = self.output_layer(feature_fusion)
        # print(f"RelightNet output shape: {output.shape}")
        return output

class LowlightEnhance(nn.Module):
    def __init__(self):
        super(LowlightEnhance, self).__init__()
        self.DecomNet_layer_num = 5
        self.DecomNet = DecomNet(layer_num=self.DecomNet_layer_num)
        self.RelightNet = RelightNet()

    def forward(self, input_low, input_high):
       #  print(f"Input low shape: {input_low.shape}, Input high shape: {input_high.shape}")
        R_low, I_low = self.DecomNet(input_low)
       #  print(f"R_low shape: {R_low.shape}, I_low shape: {I_low.shape}")
        R_high, I_high = self.DecomNet(input_high)
       #  print(f"R_high shape: {R_high.shape}, I_high shape: {I_high.shape}")

        I_delta = self.RelightNet(I_low, R_low)
        # print(f"I_delta shape: {I_delta.shape}")

        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])
        I_delta_3 = concat([I_delta, I_delta, I_delta])
       #  print(f"I_low_3 shape: {I_low_3.shape}, I_high_3 shape: {I_high_3.shape}, I_delta_3 shape: {I_delta_3.shape}")

        output_R_low = R_low
        output_I_low = I_low_3
        output_I_delta = I_delta_3
        output_S = R_low * I_delta_3
       #  print(f"Output S shape: {output_S.shape}")

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
        smooth_kernel_y = smooth_kernel_x.transpose(2,3)

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
            # Check if input_low_eval is grayscale (2D) or RGB (3D)
            if len(input_low_eval.shape) == 2:
                # Grayscale: expand dims to make it [1, height, width]
                input_low_eval = np.expand_dims(input_low_eval, axis=0)
            elif len(input_low_eval.shape) == 3:
                # RGB: transpose to [channels, height, width]
                input_low_eval = np.transpose(input_low_eval, (2, 0, 1))
            else:
                raise ValueError(f"Unexpected input shape: {input_low_eval.shape}")

            # Convert to a PyTorch tensor and add a batch dimension [batch_size, channels, height, width]
            input_low_eval = torch.from_numpy(np.expand_dims(input_low_eval, axis=0)).float()

            # Process based on the training phase
            if train_phase == "Decom":
                result_1, result_2 = self(input_low_eval, input_low_eval)[:2]
            elif train_phase == "Relight":
                result_1, result_2 = self(input_low_eval, input_low_eval)[2:]
            else:
                raise ValueError(f"Unknown training phase: {train_phase}")

            # Save the results
            print(f"result_1 shape: {result_1.shape}, result_2 shape: {result_2.shape}")
            save_images(os.path.join(sample_dir, f'eval_{train_phase}_{idx + 1}_{epoch_num}.png'), result_1, result_2)


    def train_model(self, train_low_data, train_high_data, eval_low_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr[0])

        print(f"[*] Start training for phase {train_phase}.")

        start_time = time.time()
        image_id = 0

        for epoch in range(epoch):
            self.lr=lr[epoch]
            for batch_id in range(len(train_low_data) // batch_size):
                
                # Generate data for a batch
                batch_input_low = torch.zeros(batch_size, 3, patch_size, patch_size)
                batch_input_high = torch.zeros(batch_size, 3, patch_size, patch_size)

                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)

                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = torch.from_numpy(data_augmentation(train_low_data[image_id][x:x + patch_size, y:y + patch_size, :], rand_mode).copy().transpose(2, 0, 1)).float()
                    batch_input_high[patch_id, :, :, :] = torch.from_numpy(data_augmentation(train_high_data[image_id][x:x + patch_size, y:y + patch_size, :], rand_mode).copy().transpose(2, 0, 1)).float()
                    
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



