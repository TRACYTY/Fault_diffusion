import json

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

font_size = 22
plt.rc('font', size=font_size)
plt.rc('axes', titlesize=font_size)
plt.rc('axes', labelsize=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('figure', titlesize=font_size)

import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error


def extract_time(data):
    """Returns Maximum sequence length and each sequence length."""
    time = [len(sample) for sample in data]
    max_seq_len = max(time)
    return time, max_seq_len


class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(GRUPredictor, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t):
        out, _ = self.gru(x)
        out = self.fc(out)
        y_hat = self.sigmoid(out)
        return y_hat


def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction."""

    # ori_data = torch.tensor(ori_data, dtype=torch.float32)
    # generated_data = torch.tensor(generated_data, dtype=torch.float32)

    # Basic Parameters
    no, seq_len, dim = ori_data.shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    ## Build a post-hoc RNN predictive network
    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 128

    # Build the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ori_data = ori_data.to(device)
    generated_data = generated_data.to(device)
    model = GRUPredictor(dim, hidden_dim).to(device)

    # Loss function and optimizer
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters())

    ## Training
    # Training using Synthetic dataset
    for epoch in range(iterations):
        optimizer.zero_grad()
        output = model(generated_data[:, :-1, :], torch.tensor(generated_time, dtype=torch.float32).view(-1, 1))
        loss = criterion(output, torch.roll(generated_data, -1, dims=1)[:, :-1, :])
        loss.backward()
        optimizer.step()

    ## Test the trained model on the original data
    with torch.no_grad():
        pred_Y_curr = model(ori_data[:, :-1, :], torch.tensor(ori_time, dtype=torch.float32).view(-1, 1))

    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp += mean_absolute_error(torch.roll(ori_data[i].cpu(), -1, dims=0)[:-1], pred_Y_curr[i].cpu())

    predictive_score = MAE_temp / no

    return predictive_score

def visualize_and_save_samples(ori_data, path):
    # 检查目录是否存在，如果不存在，则创建
    # path=str(args.save_dir) + "/" + save_dir
    

    import os
    if not os.path.exists(path):
        os.makedirs(path)

    # 提取前十个样本
    num_samples = 100

    import numpy as np
    ori_data = np.array(ori_data)  # 将列表转换为NumPy数组
    for i in range(min(num_samples, ori_data.shape[0])):
        plt.figure(figsize=(10, 6))
        # 绘制前三个通道
        for channel in range(min(10, ori_data.shape[2])):
            plt.plot(ori_data[i, :, channel], label=f'Channel {channel + 1}')
        plt.title(f'Sample {i + 1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        # 保存图像
        plt.savefig(path + 'sample_{}.png'.format(i + 1))
        plt.close()


def visualization(ori_data, generated_data1, analysis,path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    # Analysis sample size (for faster computation)
    anal_sample_no = min([100, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    print(idx)
    # Data preprocessing
    ori_data = np.asarray(ori_data)
    ori_data = ori_data[:, :, :-1]
    generated_data1 = np.asarray(generated_data1)

    ori_data = ori_data[idx]
    generated_data1 = generated_data1[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data1[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate(
                (prep_data, np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len]))
            )
            prep_data_hat = np.concatenate(
                (prep_data_hat, np.reshape(np.mean(generated_data1[i, :, :], 1), [1, seq_len]))
            )

    # Visualization parameter
    def check_and_clean_data(data):
        # 替换 NaN 和无穷大
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # 转换为 float32
        data = data.astype(np.float32)
        return data

    prep_data = check_and_clean_data(prep_data)
    prep_data_hat = check_and_clean_data(prep_data_hat)
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == "tsne":

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=0, perplexity=20, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(
            tsne_results[:anal_sample_no, 0],
            tsne_results[:anal_sample_no, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            tsne_results[anal_sample_no:, 0],
            tsne_results[anal_sample_no:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )
        plt.legend(prop={'size': 22}, markerscale=2)
        plt.title("t-SNE plot")
        plt.rcParams['pdf.fonttype'] = 42
        plt.savefig(path+ "tsne.png", dpi=100, bbox_inches='tight')
        plt.close()
    elif analysis == "pca":
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.show()
        # path = str(args.save_dir) + "/" + str(args.transfer_type)

        plt.savefig(path+ "_PCA.png", dpi=100, bbox_inches='tight')
        plt.close()

import scipy
import numpy as np

from Models.ts2vec.ts2vec import TS2Vec


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator
import torch.nn as nn
import torch.optim as optim

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

def discriminative_score_metrics(ori_data, generated_data, energy=False):
    # print(type(ori_data))
    # print(type(generated_data))
    # ori_data = torch.tensor(ori_data)
    no, seq_len, dim = ori_data.shape

    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    if energy:
        hidden_dim = int(dim / 3)
        iterations = 300
    else:
        hidden_dim = int(dim / 2)
        iterations = 2000
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ori_data = ori_data.to(device)
    generated_data = generated_data.to(device)
    # Build the model
    model = GRUModel(dim, hidden_dim).to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters())
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)
    train_x_combined = torch.cat([train_x, train_x_hat], dim=0)
    train_t_combined = torch.cat([torch.ones(len(train_x)), torch.zeros(len(train_x_hat))], dim=0).to(device)
    # Training
    for epoch in range(iterations):
        optimizer.zero_grad()
        output = model(train_x_combined)
        loss = criterion(output, train_t_combined.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # Testing
    test_x_combined = torch.cat([test_x, test_x_hat], dim=0)
    test_t_combined = torch.cat([torch.ones(len(test_x)), torch.zeros(len(test_x_hat))], dim=0).to(device)

    with torch.no_grad():
        output = model(test_x_combined)
        loss = criterion(output, test_t_combined.unsqueeze(1))
        predicted = (output >= 0.5).squeeze().int()
        accuracy = accuracy_score(test_t_combined.cpu().numpy(), predicted.cpu().numpy())

    discriminative_score = np.abs(0.5 - accuracy)

    return discriminative_score
def Context_FID(ori_data, generated_data):
    model = TS2Vec(input_dims=ori_data.shape[-1], device=0, batch_size=8, lr=0.001, output_dims=320,
                   max_train_length=3000)
    model.fit(ori_data, verbose=False)
    ori_represenation = model.encode(ori_data, encoding_window='full_series')
    gen_represenation = model.encode(generated_data, encoding_window='full_series')
    idx = np.random.permutation(ori_data.shape[0])
    ori_represenation = ori_represenation[idx]
    gen_represenation = gen_represenation[idx]
    results = calculate_fid(ori_represenation, gen_represenation)
    return results

data='psm'
path = "EXP_result/few_shot_exp_log0328/PSM_T1/"
import os
if not os.path.exists(path):
    os.makedirs(path)
seq_len=24
import numpy as np
import pandas as pd

def load_and_reshape(file_path):
        data = pd.read_csv(file_path).values[:, 1:26]  # 假设特征从第二列开始
        # 计算每个样本应该有的批大小和形状重塑
        batch_size = data.shape[0] // 24  # 假设每个样本包含24个时间步
        return data.reshape(batch_size, 24 , 25)  # 展平时间步和特征
    # 模拟数据
    
# class_1_ori = load_and_reshape("/home/xuyi/GT_yuan/datasets/ib600_cluster1.csv")
# real_data = np.load("/home/xuyi/GT_yuan/PSM/1/none/sys_data.npy") # 10 个真实样本，序列长度 50，特征维度 3
# generated_data = np.load("/home/xuyi/Diffusion-TS/OUTPUT/PSM/ddpm_fake_PSM.npy") # 5 个生成样本

generated_data1= np.load("./OUTPUT/PSM_T1_finetuning/samples/PSM_T1_finetuning_ground_truth_24_train.npy")
# generated_data1=np.tile(generated_data1, (5, 1, 1))
generated_data1=generated_data1[:300,:,:]
generated_data2=np.load("./OUTPUT/PSM_T1_finetuning/ddpm_fake_PSM_T1_finetuning.npy")
print(generated_data1.shape)
# generated_data2=generated_data2[:,:25]
# generated_data2=generated_data2.reshape(20, 24, 25)
# generated_data2=np.tile(generated_data2, (5, 1, 1))
print(generated_data2.shape)

print("visulization normal serises saving")
visualize_and_save_samples(generated_data1, path)
visualize_and_save_samples(generated_data2, path)
print("visulization normal PCA tsne saving")
visualization(generated_data1, generated_data2, "tsne",path)
visualization(generated_data1, generated_data2, "pca",path)
# print("generated_data1.shape",torch.tensor(generated_data1).shape)

generated_data1 = torch.tensor(generated_data1)
generated_data2 = torch.tensor(generated_data2)
###Context_FID compute
a = generated_data1.numpy()
b = generated_data2.numpy()
# path = str(args.save_dir) + "/" + str(
# args.transfer_type) + "/" + args.description + args.description_loss + "/"



# np.save(path + '/ori_data.npy', a)
# np.save(path + '/sys_data.npy', b)
import pandas as pd
df = pd.DataFrame(b.reshape(-1, b.shape[-1]))  # 将三维张量展平为二维数组，并转换为 DataFrame
# df.to_csv(path + '/sys_data.csv', index=False)
c = Context_FID(a, b)  #########
print("Context_FID compute completed:", c)
##correlation score compute
x_real = generated_data1
x_fake = generated_data2
iterations = 5
correlational_score = []
size = int(x_real.shape[0] / iterations)
from metrics.cross_correlation import CrossCorrelLoss

def random_choice(size, num_select=100):
    select_idx = np.random.randint(low=0, high=size, size=(num_select,))
    return select_idx
def display_scores(results):
   mean = np.mean(results)
   sigma = scipy.stats.sem(results)
   sigma = sigma * scipy.stats.t.ppf((1 + 0.95) / 2., 5-1)
  #  sigma = 1.96*(np.std(results)/np.sqrt(len(results)))
   print('Final Score: ', f'{mean} \xB1 {sigma}')

for i in range(iterations):
    real_idx = random_choice(x_real.shape[0], size)
    fake_idx = random_choice(x_fake.shape[0], size)
    corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
    loss = corr.compute(x_fake[fake_idx, :, :])
    correlational_score.append(loss.item())
    print(f'Iter {i}: ', 'cross-correlation =', loss.item(), '\n')
display_scores(correlational_score)
##discriminative_score compute
metric_results = dict()
metric_results['Context_FID'] = c
metric_results['correlational_score'] = correlational_score
discriminative_score = list()
generated_data1 = generated_data1.clone().detach().float()
generated_data2 = generated_data2.clone().detach().float()

for tt in range(10):
    temp_pred = discriminative_score_metrics(
        generated_data1, generated_data2, True)
    discriminative_score.append(temp_pred)

metric_results['discriminative'] = np.mean(discriminative_score)
metric_results['discriminative_std'] = np.std(discriminative_score)
print("discriminative_score compute completed")
predictive_score = list()
for tt in range(10):

    temp_pred = predictive_score_metrics(
        generated_data1, generated_data2)
    predictive_score.append(temp_pred)

metric_results['predictive'] = np.mean(predictive_score)
metric_results['predictive_std'] = np.std(predictive_score)
print("predictive_score compute completed")
print("meric:", metric_results)
file_path = path + "/metric_results.txt"
with open(file_path, "w") as file:
    file.write(json.dumps(metric_results))
print("结果已保存到 {}metric_results.txt 文件中。".format(file_path))
