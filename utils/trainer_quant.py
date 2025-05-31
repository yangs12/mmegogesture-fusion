"""Training process
"""  
import cv2
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import pickle
import seaborn as sns
from utils.result_utils import *
from utils.camera import *
import einops

class Trainer:

    def __init__(self, model_list, data_train, data_valid, data_test, args, device):
        self.model_list = model_list
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test  = data_test
        self.sensor = [args.sensor.select] if str(type(args.sensor.select))=="<class 'str'>" else args.sensor.select
        self.device = device
        self.args  = args
        self.label   = args.result.labels
        self.classes = args.result.classes
        self.path_save_vis = args.result.path_save_vis
        self.result = {}
        self.result['y']      = {}
        self.result['y_pred'] = {}

    def run_model_list(self, x_batch):
        if self.args.model.fusion == 'concat':
            radar_processed_data = einops.repeat(x_batch['rad-uD'], 'b h w -> b (copy) h w', copy=3)
            feat1 = self.model_list[0](radar_processed_data)
            feat2 = self.model_list[1](x_batch['cam-img'])
            fused = torch.cat([feat1, feat2], dim=1)
            y_batch_pred = self.model_list[-1](fused)

        elif self.args.model.fusion == 'camonly':
            feat1 = self.model_list[0](x_batch['cam-img'])
            y_batch_pred = self.model_list[-1](feat1)

        elif self.args.model.fusion == 'late':
            radar_processed_data = einops.repeat(x_batch['rad-uD'], 'b h w -> b (copy) h w', copy=3)
            output1 = self.model_list[0](radar_processed_data)
            output2 = self.model_list[1](x_batch['cam-img'])
            y_batch_pred = (output1 + output2) / 2.0

        return y_batch_pred

    def calibrate_model(self, total_batches=20):
        """
        Calibrate all models in model_list using data_train.
        Ensures all quantized submodules observe representative activations.
        """
        print(f'🔧 Starting calibration over {total_batches} batches...')
        
        for model in self.model_list:
            model.eval().to(self.device)

        loss_fn = nn.CrossEntropyLoss().to(self.device)

        num_batches = 0
        total_loss = 0.0
        progress_bar = tqdm.tqdm(self.data_train)

        with torch.no_grad():
            for iter, batch in enumerate(progress_bar):
                x_batch = {}
                for sensor_idx, sensor_sel in enumerate(self.sensor):
                    x_batch[sensor_sel] = batch[sensor_idx].to(self.device, dtype=torch.float)
                y_batch = batch[-2].to(self.device, dtype=torch.long)

                if self.args.model.fusion == 'concat':
                    radar_input = einops.repeat(x_batch['rad-uD'], 'b h w -> b c h w', c=3)

                    # Trigger backbone observers
                    _ = self.model_list[0](radar_input)
                    _ = self.model_list[1](x_batch['cam-img'])

                    # Trigger classifier observers
                    feat1 = self.model_list[0](radar_input)
                    feat2 = self.model_list[1](x_batch['cam-img'])
                    fused = torch.cat([feat1, feat2], dim=1)
                    y_pred = self.model_list[2](fused)

                elif self.args.model.fusion == 'camonly':
                    _ = self.model_list[0](x_batch['cam-img'])

                    feat = self.model_list[0](x_batch['cam-img'])
                    y_pred = self.model_list[1](feat)

                elif self.args.model.fusion == 'late':
                    radar_input = einops.repeat(x_batch['rad-uD'], 'b h w -> b c h w', c=3)

                    _ = self.model_list[0](radar_input)
                    _ = self.model_list[1](x_batch['cam-img'])

                    output1 = self.model_list[0](radar_input)
                    output2 = self.model_list[1](x_batch['cam-img'])
                    y_pred = (output1 + output2) / 2.0  # still OK in float domain

                else:
                    raise ValueError(f"Unknown fusion type: {self.args.model.fusion}")

                loss = loss_fn(y_pred, y_batch)
                total_loss += loss.item()
                num_batches += 1

                if num_batches % 5 == 0:
                    print(f'  [Batch {num_batches}] Loss: {loss.item():.4f}')

                if num_batches >= total_batches:
                    break

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f'✅ Calibration complete. Avg loss over {num_batches} batches: {avg_loss:.4f}')

    def train(self):
        # turn all models to training
        listed_params = []
        for model in self.model_list:
            model.train().to(self.device)
            listed_params += list(model.parameters())
        optimizer = torch.optim.AdamW(listed_params,
                                        lr=self.args.train.learning_rate, weight_decay=self.args.train.weight_decay)
        loss_fn=nn.CrossEntropyLoss().to(self.device)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                T_0 = 25,# Number of iterations for the first restart
                                                T_mult = 1, # A factor increases TiTi​ after a restart
                                                eta_min = 0.01*self.args.train.learning_rate) # Minimum learning rate

        Epoch_num = self.args.train.epoch
        print("Total Epochs: ", Epoch_num)
        test_loss_best = 100
        test_acc_best  = 0
        step = 0
        for epoch in range(Epoch_num):
            train_loss = 0
            train_num  = 0
            progress_bar = tqdm.tqdm(self.data_train)
            for iter, data in enumerate(progress_bar):
                x_batch = {}
                for sensor_idx, sensor_sel in enumerate(self.sensor):
                    x_batch[sensor_sel] = data[sensor_idx].to(self.device, dtype=torch.float)
                y_batch   = data[-2].to(self.device, dtype=torch.long)
                des_batch = data[-1]

                y_batch_pred = self.run_model_list(x_batch)
                
                loss = loss_fn(y_batch_pred, y_batch)

                train_num += len(y_batch)              
                train_loss += loss.item() * len(y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print
                step += 1
                progress_bar.set_description(
                'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Total loss: {:.5f}.'.format(
                    step, epoch+1, Epoch_num, iter + 1, len(self.data_train), train_loss/train_num))
            test_acc, test_loss, test_y, test_y_pred, test_y_prob, test_des = self.test(self.data_test, self.device, self.model_list, loss_fn, epoch)
            lr_scheduler.step()
            # if epoch==0:
            #     GFlops, N_params = self.cal_model_stats(self.model, self.data_test)
            self.result['y'][epoch]      = test_y
            self.result['y_pred'][epoch] = test_y_pred
            print('test acc ', test_acc, 'test_loss ', test_loss)
            if self.args.wandb.use_wandb:
                wandb.log({
                    'lr': lr_scheduler.optimizer.param_groups[0]['lr'],
                    'train_loss': train_loss/train_num,
                    'test_acc': test_acc, 
                    'test_loss':  test_loss,
                    # 'GFLOPs': GFlops,
                    # 'N_param': N_params
                    }, step=epoch)
                if self.args.wandb.log_all:
                    wandb.log({
                        f'roc': wandb.plot.roc_curve(test_y, test_y_prob, labels=list(self.classes), title='ROC'),
                        f'consufion': wandb.plot.confusion_matrix(y_true=test_y, preds=test_y_pred, class_names=list(self.classes), title='Confusion')
                        }, step=epoch)
                
            if test_acc>=test_acc_best:
                test_acc_best = test_acc
                if self.args.result.save_vis and epoch>0:
                    save_result_confusion(test_y, test_y_pred, self.label, 'best-'+self.args.result.name, self.path_save_vis)
                    save_result_statistics(test_y, test_y_pred, test_des, 'best-'+self.args.result.name, self.path_save_vis)
                # save the best model
                if self.args.result.save_model:
                    if not os.path.exists(self.args.result.path_save_model):
                        os.makedirs(self.args.result.path_save_model)
                    for i,model in enumerate(self.model_list):
                        model_save_path = os.path.join(self.args.result.path_save_model, self.args.result.name+'_'+str(i)+'_best.pt')
                        torch.save(model.state_dict(), model_save_path)
            if test_loss<=test_loss_best:
                test_loss_best = test_loss
                    
        if self.args.result.save_vis:
            save_result_confusion(test_y, test_y_pred, self.label, 'last-'+self.args.result.name, self.path_save_vis)
            save_result_statistics(test_y, test_y_pred, test_des, 'last-'+self.args.result.name, self.path_save_vis)
            save_result_pred(self.result, 'last-'+self.args.result.name, self.path_save_vis)
        
        # save the last model
        if self.args.result.save_model:
            if not os.path.exists(self.args.result.path_save_model):
                os.makedirs(self.args.result.path_save_model)
            for i,model in enumerate(self.model_list):
                model_save_path = os.path.join(self.args.result.path_save_model, self.args.result.name+'_'+str(i)+'_last.pt')
                torch.save(model.state_dict(), model_save_path)  

    def test(self, data_test, device, model_list, loss_fn, epoch):
        test_loss = 0.0
        correct, total = 0, 0
        y_pred = []
        y_prob = []
        y_true = []
        des    = []
        # ✨ W&B: Create a Table to store predictions for each test step 
        # (ref: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb#scrollTo=J-PiBiE1LNiE)
        if self.args.wandb.use_wandb: 
            columns=["id"]
            for sensor_sel in self.sensor:
                columns.append("input_" + str(sensor_sel))
            columns.append('gt')
            columns.append('pred')
            for class_name in self.classes:
                columns.append(str(class_name))
            test_table = wandb.Table(columns=columns)

        for data in data_test:
            with torch.no_grad():
                x_batch = {}
                for sensor_idx, sensor_sel in enumerate(self.sensor):
                    x_batch[sensor_sel] = data[sensor_idx].to(self.device, dtype=torch.float)
                y_batch = data[-2].to(self.device, dtype=torch.long)
                des_batch = data[-1]

                for model in model_list:
                    model.eval().to(self.device)

                y_batch_prob = self.run_model_list(x_batch)

                loss = loss_fn(y_batch_prob, y_batch)
                y_batch_pred = torch.argmax(y_batch_prob,axis = 1)
                correct += torch.sum(y_batch_pred==y_batch)
                total += len(y_batch)              
                test_loss += loss.item() * x_batch[self.sensor[0]].size(0)
                if self.args.wandb.use_wandb:
                    self.log_test_predictions(des_batch, x_batch, y_batch, 
                                                y_batch_prob, y_batch_pred, test_table, NUM_IMAGES_PER_BATCH=1)

                for i in range(x_batch[self.sensor[0]].size(0)):
                    y_pred.append(y_batch_pred[i].item())
                    y_true.append(y_batch[i].item())
                    y_prob.append(y_batch_prob[i].to('cpu').numpy())
                    des.append(des_batch[i])
        acc = correct/total
        y_prob = [list(sample) for sample in y_prob]
        if self.args.wandb.use_wandb:
            if self.args.wandb.log_all:
                wandb.log({"test_preds" : test_table}, step=epoch)
        return acc.item(), test_loss/total, y_true, y_pred, y_prob, des
    
    def cal_model_stats(self, model, data_test, flag_table=False):
        """
        Output: Flops (G), # params (M)
        """
        from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table
        dat_sample   = [data_test.dataset[0][0].unsqueeze(dim=0)]
        x = {}
        model.eval()
        for sensor_idx, sensor_sel in enumerate(self.sensor):
            x[sensor_sel] = dat_sample[sensor_idx].to(self.device, dtype=torch.float)
        flops = FlopCountAnalysis(model, x)
        param = parameter_count(model)
        if flag_table:
            print(parameter_count_table(model))
        return flops.total()/1e9, param[list(param)[1]]/1e6
    
    def log_test_predictions(self, des, inputs, labels, outputs, predicted, test_table, NUM_IMAGES_PER_BATCH):
        # obtain confidence scores for all classes
        scores = F.softmax(outputs.data, dim=1)
        log_scores = scores.cpu().numpy()
        # log_images = images.cpu().numpy()
        log_labels = labels.cpu().numpy()
        log_preds = predicted.cpu().numpy()
        # adding ids based on the order of the images
        for i in range(len(des)):
            # add required info to data table:
            # id, image pixels, model's guess, true label, scores for all classes
            if i == NUM_IMAGES_PER_BATCH:
                break
            data_sel = []
            d, l, p, s = des[i], log_labels[i], log_preds[i], log_scores[i]
            img_id = str(d['Episode']) + "_" +  str(d['Order'])
            data_sel.append(img_id)
            for sensor_sel in self.sensor:
                if 'rad' in sensor_sel:
                    data_sel.append(wandb.Image(inputs[sensor_sel][i].cpu().numpy()))
                elif 'cam' in sensor_sel:
                    img = inputs[sensor_sel][i]
                    img = img if 'img' in sensor_sel else img[:,img.shape[1]//2,:,:]
                    data_sel.append(wandb.Image(img.permute(1,2,0).cpu().numpy()))
            data_sel.extend((self.classes[int(l)], self.classes[int(p)], *s))
            test_table.add_data(*data_sel)
    
            