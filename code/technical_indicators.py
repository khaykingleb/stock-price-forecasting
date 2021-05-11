import pandas as pd

def simple_moving_average(sequence: pd.Series, n: int) -> pd.Series:
  """
  Calculate simple n-day moving average for given data.

  Params:
      
  Returns:
  """
  return sequence.rolling(window=n).mean()


def weighted_moving_average(sequence: pd.Series, n: int) -> pd.Series:
  """
  Calculate weighted n-day moving average for given data.

  Params:
      
  Returns:
  """
  return sequence.rolling(window=n).apply(lambda x: x[::-1].cumsum().sum() * 2 / n / (n + 1))


def exponential_moving_average(sequence: pd.Series, n: int) -> pd.Series:
    """
    Calculate exponential n-day moving average for given data.

    Params:
        
    Returns:
    """
    return sequence.ewm(span=n).mean()
  


