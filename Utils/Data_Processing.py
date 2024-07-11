import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, BatchSampler, DataLoader

# historical max stock price on 07/10/2024 was 4.033e+03
#  the max value representable by float16 is 6.550e+04
STOCK_PRICE_DTYPE = np.float16
STOCK_PRICE_DTYPE_TORCH = torch.float16
STOCK_DT_DTYPE = np.float16 # np.uint8
STOCK_DT_DTYPE_TORCH = torch.float16 # torch.uint8

def split_past_fut(A, DT, T_PAST):
  return A[:, :T_PAST], A[:, T_PAST:], DT[:, :T_PAST], DT[:, T_PAST:]


def scale_by_1st_col(A):
  return A / A[:, 0:1], A[:, 0:1]

def scale_by_col(A, c):
  return A / c, c


def identity_scale(A):
  return A, np.ones((A.shape[0], 1))


class StockDataset(Dataset):
  """
  Dataset for individual stocks, should only need to be called by `make_market_dataset`
  """
  def __init__(self, stock_ticker, window_path, window_dt_path, t_past, t_fut, num_windows=1, scale=False):
    self.t_past = t_past
    self.num_windows = num_windows
    # read the stock data from the file
    stock_windows = np.load(os.path.join(window_path, f'{stock_ticker}_windows.npy')).astype(STOCK_PRICE_DTYPE)
    # read the delta times between stock observations from a file, or default it to 1 day b/w observations
    dt_file_path = os.path.join(window_dt_path, f'{stock_ticker}_windows_dt.npy')

    stock_windows_dt = np.load(dt_file_path).astype(STOCK_DT_DTYPE) if os.path.exists(dt_file_path) else \
      np.ones_like(stock_windows, dtype=STOCK_DT_DTYPE)

    assert stock_windows.shape[0] == stock_windows_dt.shape[0], \
      'different # of samples in stock_windows and stock_windows_dt'
    assert stock_windows.shape[1] == stock_windows_dt.shape[1], \
      'different # of colums in stock_windows and stock_windows_dt'
    assert stock_windows.shape[1] == t_past + t_fut, '# columns in stock_windows != t_past + t_future'

    # append `num_windows`-1 rows of junk data to stock_windows, stock_windows_dt to handle incomplete windows
    self.n = stock_windows.shape[0]
    stock_windows = np.concatenate((stock_windows, np.ones((num_windows-1, stock_windows.shape[1]))))
    stock_windows_dt = np.concatenate((stock_windows_dt, np.ones((num_windows-1, stock_windows_dt.shape[1]))))

    # separate the first `t_past` and next `t_fut` cols of the files
    stock_past, stock_fut, self.dt_past, self.dt_fut = split_past_fut(stock_windows, stock_windows_dt, t_past)
    # optionally scale the past and future stock data by the first value of each observation
    self.std_past, self.past_scales = scale_by_1st_col(stock_past)
    self.std_fut, self.fut_scales = scale_by_col(stock_fut, stock_past[:, -1:])

  def __len__(self):
    return self.n

  def __getitem__(self, idx):
    mask = np.zeros(self.num_windows, dtype=np.bool_)
    # the last few (up to `self.num_windows`-1) rows might be padding
    if idx > self.n - self.num_windows:
      mask[-(idx-self.n+self.num_windows):] = True

    return (
      self.std_past[idx:idx+self.num_windows].astype(STOCK_PRICE_DTYPE),
      self.past_scales[idx:idx+self.num_windows].astype(STOCK_PRICE_DTYPE),
      self.dt_past[idx:idx+self.num_windows].astype(STOCK_DT_DTYPE),
      self.std_fut[idx:idx+self.num_windows].astype(STOCK_PRICE_DTYPE),
      self.fut_scales[idx:idx+self.num_windows].astype(STOCK_PRICE_DTYPE),
      self.dt_fut[idx:idx+self.num_windows].astype(STOCK_DT_DTYPE),
      mask
    )


def make_market_dataset(stock_tickers, window_path, window_dt_path, t_past, t_fut, num_windows=1, scale=False):
  """
  makes a Pytorch Dataset using stocks in stock_tickers.
  each sample is a tuple of:
  -  past_data (num_windows, t_past) ~ scaled past stock data
  -  past scales (num_windows, ) ~ the scale factor calculated from and applied to past stock data
  -  past_dt (num_windows, t_past) ~ the number of days in between observations of past stock data
  -  fut_data (num_windows, t_fut) ~ scaled future stock data
  -  fut scales (num_windows, ) ~ the scale factor calculated from and applied to future stock data
  -  fut_dt (num_windows, t_fut) ~ the number of days in between observations of future stock data
  -  mask (num_windows, ) ~ if mask[i], then the i'th sample is masked

  :param stock_tickers: a list of strings
  :param window_path: the path of the stock price data
  :param window_dt_path: the path of the delta time data
  :param t_past: the number of past datapoints
  :param t_fut: the number of future datapoints
  :param num_windows: the number of consecutive windows of prices, delta times. defaults to 1
  :param scale: if True, then Utils.Data_Processing.scale_by_1st_col is called
  :return: a Pytorch ConcatDataset
  """
  datasets = [None] * len(stock_tickers)
  if not window_dt_path:
    window_dt_path = ''

  for i, stock_ticker in enumerate(stock_tickers):
    datasets[i] = StockDataset(stock_ticker, window_path, window_dt_path, t_past, t_fut, num_windows, scale)

  return ConcatDataset(datasets)


def make_market_dataloader(market_dataset, batch_size, num_workers=0):
  # ds_indices = market_dataset.cumulative_sizes
  #
  # batch_sampler = BatchSampler(ds_indices, batch_size, drop_last=True)
  # return DataLoader(market_dataset, batch_sampler=batch_sampler, num_workers=num_workers)

  return DataLoader(market_dataset, batch_size, True, num_workers=num_workers, drop_last=True)