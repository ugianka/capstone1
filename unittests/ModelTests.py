import unittest
import sys
from os.path import join, exists
import os
from model import model_train, model_predict, MODEL_VERSION, MODEL_DIR

sys.path.insert(1, '../src')
sys.path.insert(2, '..')
sys.path.insert(3, '../unittest')


class ModelTest(unittest.TestCase):
    def test_train_file_creation(self):
        data_dir = join('.', 'data')
        work_dir = join(data_dir, 'work-data')
        models_dir = join('.', 'models')
        inpfile = join(work_dir, 'train-data-cleaned.csv')
        force_data_load = True
        if exists(inpfile):
            force_data_load = False
        model_train(data_dir=data_dir, test=True, model_dir=models_dir,
                    force_data_load=force_data_load)
        outfile = join(work_dir, 'test-all-0_1')
        return exists(outfile)

    def test_predict_result_is_numeric(self):
        data_dir = join('.', 'data')
        models_dir = join('.', 'models')
        work_dir = join(data_dir, 'work-data')
        inpfile = join(work_dir, 'train-data-cleaned.csv')
        force_data_load = True
        if exists(inpfile):
            force_data_load = False

        ukmodelfile = join(models_dir, 'test-united_kingdom-' +
                           MODEL_VERSION + '.joblib')
        if not exists(ukmodelfile):
            model_train(data_dir=data_dir, model_dir=models_dir, test=True,
                        force_data_load=force_data_load)
        pred = model_predict('united_kingdom', '2018', '1', '1', data_dir=data_dir,
                             model_dir=models_dir, test=True)
        # print(pred)
        return pred['y_pred'] > 0
