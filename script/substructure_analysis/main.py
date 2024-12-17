from SMEG_model_hyperopt import SMEG_hyperopt
from maskgnn import set_random_seed
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    set_random_seed(10)
    classification_task_list = ['Mutagenicity', 'hERG', 'cyp3a4', 'cyp2c19']
    for task in classification_task_list:
        SMEG_hyperopt(10, task, 30, classification=True)