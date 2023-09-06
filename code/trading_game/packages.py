########################################################################
######################## Import Packages
########################################################################

import numpy as np
import pandas as pd
from numba import jit
import random
import matplotlib.pyplot as plt
from pprint import pprint
from IPython.display import display
from tabulate import tabulate
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10,8) 
plt.style.use('ggplot')