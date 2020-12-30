import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from model import U_Net,R2AttU_Net
import csv
import pandas as pd

def jaccard_loss(true, logits, eps=1e-7):
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1).to(torch.int64)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()
  

    
	def build_model(self):
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=3,output_ch=1)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=3,output_ch=1)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)
			

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)


	def print_network(self, model, name):
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img
	


	def train(self):

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
		train_history_path = os.path.join(self.model_path, 'train-%s-%d-%.4f-%d-%.4f.csv' % (self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
		valid_history_path = os.path.join(self.model_path, 'valid-%s-%d-%.4f-%d-%.4f.csv' % (self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
		if os.path.isfile(unet_path):
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			lr = self.lr
			best_unet_score = 0.
			train_history = []
			valid_history = []
			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length = 0

				for i, (images, GT) in enumerate(self.train_loader):
					images = images.to(self.device)
					GT = GT.to(self.device)


					SR = self.unet(images)
					SR_probs = F.sigmoid(SR)
					SR_flat = SR_probs.view(SR_probs.size(0),-1)
					GT_flat = GT.view(GT.size(0),-1)
					loss = self.criterion(SR_flat,GT_flat)
					epoch_loss += loss.item()

					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)
					length += images.size(0)

				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length


				print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss,\
					  acc, SE, SP, PC, F1, JS, DC))
				train_history.append([acc, SE, SP, PC, F1, JS, DC])

				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				
				
				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length=0
				for i, (images, GT) in enumerate(self.valid_loader):

					images = images.to(self.device)
					GT = GT.to(self.device)
					SR = F.sigmoid(self.unet(images))
					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)
						
					length += images.size(0)
					
				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				unet_score = JS + DC

				print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))
				valid_history.append([acc, SE, SP, PC, F1, JS, DC])
				'''
				torchvision.utils.save_image(images.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(SR.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(GT.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
				'''


				if unet_score > best_unet_score:
					best_unet_score = unet_score
					best_epoch = epoch
					best_unet = self.unet.state_dict()
					print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
					torch.save(best_unet,unet_path)
			train_history = pd.DateFrame(train_history, columns=['acc', 'SE', 'SP', 'PC', 'F1', 'JS', 'DC'])
			valid_history = pd.DateFrame(valid_history, columns=['acc', 'SE', 'SP', 'PC', 'F1', 'JS', 'DC'])
			train_history.to_csv(train_history_path)
			valid_history.to_csv(valid_history_path)

			#===================================== Test ====================================#
			del self.unet
			del best_unet
			self.build_model()
			self.unet.load_state_dict(torch.load(unet_path))
			
			self.unet.train(False)
			self.unet.eval()

			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length=0
			for i, (images, GT) in enumerate(self.valid_loader):

				images = images.to(self.device)
				GT = GT.to(self.device)
				SR = F.sigmoid(self.unet(images))
				acc += get_accuracy(SR,GT)
				SE += get_sensitivity(SR,GT)
				SP += get_specificity(SR,GT)
				PC += get_precision(SR,GT)
				F1 += get_F1(SR,GT)
				JS += get_JS(SR,GT)
				DC += get_DC(SR,GT)
						
				length += images.size(0)
					
			acc = acc/length
			SE = SE/length
			SP = SP/length
			PC = PC/length
			F1 = F1/length
			JS = JS/length
			DC = DC/length
			unet_score = JS + DC


			f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
			f.close()
			

			
