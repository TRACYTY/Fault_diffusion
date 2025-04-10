## Necessary Packages
import numpy as np
import tensorflow as tf
import torch
import torch
import numpy as np


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
  """Divide train and test data for both original and synthetic data.

  Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  data_t = torch.tensor(data_t, dtype=torch.int64)
  data_t_hat = torch.tensor(data_t_hat, dtype=torch.int64)

  no = data_x.size(0)
  idx = torch.randperm(no)
  train_idx = idx[:int(no * train_rate)]
  test_idx = idx[int(no * train_rate):]

  train_x = data_x[train_idx]
  test_x = data_x[test_idx]
  train_t = data_t[train_idx]
  test_t = data_t[test_idx]

  # Divide train/test index (synthetic data)
  no = data_x_hat.size(0)
  idx = torch.randperm(no)
  train_idx = idx[:int(no * train_rate)]
  test_idx = idx[int(no * train_rate):]

  train_x_hat = data_x_hat[train_idx]
  test_x_hat = data_x_hat[test_idx]
  train_t_hat = data_t_hat[train_idx]
  test_t_hat = data_t_hat[test_idx]

  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


# def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
#   """Divide train and test data for both original and synthetic data.
#
#   Args:
#     - data_x: original data
#     - data_x_hat: generated data
#     - data_t: original time
#     - data_t_hat: generated time
#     - train_rate: ratio of training data from the original data
#   """
#   # Divide train/test index (original data)
#   no = len(data_x)
#   idx = np.random.permutation(no)
#   train_idx = idx[:int(no*train_rate)]
#   test_idx = idx[int(no*train_rate):]
#
#   train_x = [data_x[i] for i in train_idx]
#   test_x = [data_x[i] for i in test_idx]
#   train_t = [data_t[i] for i in train_idx]
#   test_t = [data_t[i] for i in test_idx]
#
#   # Divide train/test index (synthetic data)
#   no = len(data_x_hat)
#   idx = np.random.permutation(no)
#   train_idx = idx[:int(no*train_rate)]
#   test_idx = idx[int(no*train_rate):]
#
#   train_x_hat =[data_x_hat[i] for i in train_idx]
#   test_x_hat = [data_x_hat[i] for i in test_idx]
#   train_t_hat =[data_t_hat[i] for i in train_idx]
#   test_t_hat = [data_t_hat[i] for i in test_idx]
#
#   return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat
#

def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len


def rnn_cell(module_name, hidden_dim):
  """Basic RNN Cell.
    
  Args:
    - module_name: gru, lstm, or lstmLN
    
  Returns:
    - rnn_cell: RNN Cell
  """
  assert module_name in ['gru','lstm','lstmLN']
  
  # GRU
  if (module_name == 'gru'):
    rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM
  elif (module_name == 'lstm'):
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM Layer Normalization
  elif (module_name == 'lstmLN'):
    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  return rnn_cell


def random_generator (batch_size, z_dim, T_mb, max_seq_len):
  """Random vector generation.
  
  Args:
    - batch_size: size of the random vector
    - z_dim: dimension of random vector
    - T_mb: time information for the random vector
    - max_seq_len: maximum sequence length
    
  Returns:
    - Z_mb: generated random vector
  """
  Z_mb = list()
  for i in range(batch_size):
    temp = np.zeros([max_seq_len, z_dim])
    temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
    temp[:T_mb[i],:] = temp_Z
    Z_mb.append(temp_Z)
  return Z_mb


def batch_generator(data, time, batch_size):
  """Mini-batch generator.
  
  Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch
    
  Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
  """
  no = len(data)
  idx = np.random.permutation(no)
  train_idx = idx[:batch_size]     
            
  X_mb = list(data[i] for i in train_idx)
  T_mb = list(time[i] for i in train_idx)
  
  return X_mb, T_mb