import pandas as pd
import os

task_list = ['Mutagenicity', 'hERG', 'cyp3a4', 'cyp2c19']   
sub_type_list = ['mol']     # 'mol', 'brics', 'fg', 'murcko'

for task_name in task_list:
    for sub_type in sub_type_list:
        try:
            print('{} {} sum succeed.'.format(task_name, sub_type))
           
            result_summary = pd.DataFrame()
            for i in range(10):
                seed = i + 1
                result_train = pd.read_csv('../../outputs/substructure_analysis/{}/{}_{}_{}_train_prediction.csv'.format(sub_type, task_name, sub_type, seed))
                result_val = pd.read_csv('../../outputs/substructure_analysis/{}/{}_{}_{}_val_prediction.csv'.format(sub_type, task_name, sub_type, seed))
                result_test = pd.read_csv('../../outputs/substructure_analysis/{}/{}_{}_{}_test_prediction.csv'.format(sub_type, task_name, sub_type, seed))

                group_list = ['training' for x in range(len(result_train))] + ['val' for x in range(len(result_val))] + ['test' for
                                                                                                                         x in range(
                        len(result_test))]
                result = pd.concat([result_train, result_val, result_test], axis=0)
                
                result['group'] = group_list
                if sub_type == 'mol':
                    result.sort_values(by='smiles', inplace=True)
                
                if seed == 1:
                    result_summary['smiles'] = result['smiles']
                    result_summary['label'] = result['label']
                    result_summary['sub_name'] = result['sub_name']
                    result_summary['group'] = result['group']
                    result_summary['pred_{}'.format(seed)] = result['pred'].tolist()
                if seed > 1:
                    result_summary['pred_{}'.format(seed)] = result['pred'].tolist()
            pred_columnms = ['pred_{}'.format(i+1) for i in range(10)]
            data_pred = result_summary[pred_columnms]
            result_summary['pred_mean'] = data_pred.mean(axis=1)
            result_summary['pred_std'] = data_pred.std(axis=1)
            dirs = '../../outputs/substructure_analysis/summary/'
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            result_summary.to_csv('../../outputs/substructure_analysis/summary/{}_{}_prediction_summary.csv'.format(task_name, sub_type), index=False)
        except:
            print('{} {} sum failed.'.format(task_name, sub_type))