import numpy as np

def split_past_fut(A, DT, T_PAST):
  return A[:, :T_PAST], A[:, T_PAST:], DT[:, :T_PAST], DT[:, T_PAST:]

def scale_by_1st_col(A):
  return A/A[:, 0:1], A[:, 0:1]