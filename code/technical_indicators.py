import pandas as pd

def simple_moving_average(prices: pd.Series, n: int) -> pd.Series:
  """
  Calculate simple n-day moving average for given data.

  Params:
      
  Returns:
  """
  return prices.rolling(window=n).mean()


def weighted_moving_average(prices: pd.Series, n: int) -> pd.Series:
  """
  Calculate weighted n-day moving average for given data.

  Params:
      
  Returns:
  """
  return prices.rolling(window=n).apply(lambda x: x[::-1].cumsum().sum() * 2 / n / (n + 1))


def exponential_moving_average(prices: pd.Series, n: int) -> pd.Series:
    """
    Calculate exponential n-day moving average for given data.

    Params:
        
    Returns:
    """
    return prices.ewm(span=n).mean()
  
  
def relative_strength_index(prices: pd.Series, n: int) -> pd.Series:
  """
  Calculate n-day relative strength index for given data.

  Params:

  Returns:
  """
  deltas = prices.diff()
  ups = deltas.clip(lower=0)
  downs = (-deltas).clip(lower=0)
  rs = ups.ewm(com=n-1, min_periods=n).mean() / downs.ewm(com=n-1, min_periods=n).mean()

  return 100 - 100 / (1 + rs)
    

 

