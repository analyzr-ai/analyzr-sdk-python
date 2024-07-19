"""
Copyright (c) 2024 Go2Market Insights, Inc
All rights reserved.
https://g2m.ai

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import unittest, json, math, sys, uuid
import pandas as pd
from analyzrclient import Analyzer
from .utils import *

CLIENT_ID = 'test'
N_FEATURES = 2
N_SAMPLES = 1000
EPSILON = 1e-4
VERBOSE = False

with open('tests/config.json') as json_file: config = json.load(json_file)
analyzer = Analyzer(host=config['host'])
analyzer.login()
analyzer.propensity.buffer_purge(client_id=CLIENT_ID) # need to make sure buffer is empty before we start


class CommonTest(unittest.TestCase):

    def test_version(self):
        res = analyzer.version()
        self.assertEqual(res['api']['status'], 200)

class BufferTest(unittest.TestCase):

    def test_staging_single_batch_compressed_staging(self):
        df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/titanic.csv', encoding = "ISO-8859-1", low_memory=False)
        request_id = str(uuid.uuid4())
        res = analyzer.cluster._buffer_save(df, batch_size=1000, client_id=CLIENT_ID, request_id=request_id, compressed=True, staging=True)
        self.assertEqual(res['batches_saved'], 1)
        df2 = analyzer.cluster._buffer_read(client_id=CLIENT_ID, request_id=request_id, staging=True, dataframe_name='obj')
        self.assertEqual(df.shape==df2.shape, True)
        self.assertEqual(len(df.columns), len(df2.columns))
        for i in range(0, len(df.columns)): self.assertEqual(df.columns[i], df2.columns[i])
        self.assertEqual(df.loc[890, 'Name']==df2.loc[890, 'Name'], True)
        res = analyzer.cluster._buffer_clear(client_id=CLIENT_ID, request_id=request_id)
        self.assertEqual(res['status'], 200)
        self.assertEqual(res['response']['message'], 'Buffer cleared')

    def test_staging_single_batch_compressed_no_staging(self):
        df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/titanic.csv', encoding = "ISO-8859-1", low_memory=False)
        request_id = str(uuid.uuid4())
        res = analyzer.cluster._buffer_save(df, batch_size=1000, client_id=CLIENT_ID, request_id=request_id, compressed=True, staging=False)
        self.assertEqual(res['batches_saved'], 1)
        df2 = analyzer.cluster._buffer_read(client_id=CLIENT_ID, request_id=request_id, staging=False, dataframe_name='df')
        self.assertEqual(df.shape==df2.shape, True)
        self.assertEqual(len(df.columns), len(df2.columns))
        res = analyzer.cluster._buffer_clear(client_id=CLIENT_ID, request_id=request_id)
        self.assertEqual(res['status'], 200)
        self.assertEqual(res['response']['message'], 'Buffer cleared')

    def test_staging_multi_batch_compressed_staging(self):
        df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/titanic.csv', encoding = "ISO-8859-1", low_memory=False)
        request_id = str(uuid.uuid4())
        res = analyzer.cluster._buffer_save(df, batch_size=500, client_id=CLIENT_ID, request_id=request_id, compressed=True, staging=True)
        self.assertEqual(res['batches_saved'], 2)
        df2 = analyzer.cluster._buffer_read(client_id=CLIENT_ID, request_id=request_id, staging=True, dataframe_name='obj')
        self.assertEqual(df.shape==df2.shape, True)
        self.assertEqual(len(df.columns), len(df2.columns))
        for i in range(0, len(df.columns)): self.assertEqual(df.columns[i], df2.columns[i])
        self.assertEqual(df.loc[890, 'Name']==df2.loc[890, 'Name'], True)
        res = analyzer.cluster._buffer_clear(client_id=CLIENT_ID, request_id=request_id)
        self.assertEqual(res['status'], 200)
        self.assertEqual(res['response']['message'], 'Buffer cleared')

    def test_staging_multi_batch_compressed_no_staging(self):
        df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/titanic.csv', encoding = "ISO-8859-1", low_memory=False)
        request_id = str(uuid.uuid4())
        res = analyzer.cluster._buffer_save(df, batch_size=500, client_id=CLIENT_ID, request_id=request_id, compressed=True, staging=False)
        self.assertEqual(res['batches_saved'], 2)
        df2 = analyzer.cluster._buffer_read(client_id=CLIENT_ID, request_id=request_id, staging=False, dataframe_name='df')
        self.assertEqual(df.shape==df2.shape, True)
        self.assertEqual(len(df.columns), len(df2.columns))
        res = analyzer.cluster._buffer_clear(client_id=CLIENT_ID, request_id=request_id)
        self.assertEqual(res['status'], 200)
        self.assertEqual(res['response']['message'], 'Buffer cleared')

class TasksTest(unittest.TestCase):

    def test_task_simple(self):
        res = analyzer.test.run(type='simple', verbose=VERBOSE)
        self.assertEqual(res['status'], 200)

    def test_task_storage(self):
        res = analyzer.test.run(type='storage', verbose=VERBOSE)
        self.assertEqual(res['status'], 200)

class ClusteringTest(unittest.TestCase):

    def test_birch(self):
        df = load_titanic_dataset()
        res = analyzer.cluster.run(df, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='birch', buffer_batch_size=1000, cluster_batch_size=150, verbose=VERBOSE)
        df2 = res['data']
        model_id = res['request_id']
        res = analyzer.cluster.buffer_usage(client_id=CLIENT_ID)
        self.assertEqual(df2.iloc[0]['Fare'], 7.25)
        # self.assertEqual(df2.iloc[0]['PC_ID'], 3)
        self.assertEqual(df2.iloc[711]['Embarked'], 'Q')
        # self.assertEqual(df2.iloc[711]['PC_ID'], 3)
        self.assertEqual(res['response']['n_rows'], 0)
        res = analyzer.cluster.predict(df, model_id=model_id, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            buffer_batch_size=1000, verbose=VERBOSE)
        df3 = res['data2']
        self.assertEqual(len(df3), 712)
        self.assertEqual(df3.columns[0], 'PC_ID')

    def test_larger_dataset(self):
        df = load_banking_dataset()
        res = analyzer.cluster.run(df, client_id=CLIENT_ID, idx_var='index',
            categorical_vars=['y', 'housing', 'education', 'marital', 'job'], 
            numerical_vars=['campaign', 'duration', 'balance', 'loan', 'default', 'age'],
            algorithm='pca-kmeans-simple', buffer_batch_size=100000, verbose=VERBOSE)
        df2 = res['data']
        model_id = res['request_id']
        res = analyzer.cluster.buffer_usage(client_id=CLIENT_ID)
        self.assertEqual(df2.iloc[0]['duration'], 261)
        self.assertEqual(df2.iloc[45210]['job'], 'entrepreneur')
        self.assertEqual(res['response']['n_rows'], 0)
        res = analyzer.cluster.predict(df, model_id=model_id, client_id=CLIENT_ID, idx_var='index',
            categorical_vars=['y', 'housing', 'education', 'marital', 'job'], 
            numerical_vars=['campaign', 'duration', 'balance', 'loan', 'default', 'age'],
            buffer_batch_size=100000, verbose=VERBOSE)
        df3 = res['data2']
        self.assertEqual(len(df3), 45211)
        self.assertEqual(df3.columns[0], 'PC_ID')

class PropensityTest(unittest.TestCase):

    def test_logistic_regression_classifier(self):
        df = load_titanic_dataset()
        res = analyzer.propensity.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Survived', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='logistic-regression-classifier', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
        model_id = res['model_id']
        self.assertEqual(len(res['features']), 10)
        self.assertEqual(res['confusion_matrix'].shape, (2, 2))
        self.assertEqual(len(res['stats']), 7)
        res = analyzer.propensity.predict(df, model_id=model_id, client_id=CLIENT_ID,
            idx_var='PassengerId', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            buffer_batch_size=1000, verbose=VERBOSE)
        res2 = analyzer.propensity.buffer_usage(client_id=CLIENT_ID)
        df3 = res['data2']
        self.assertEqual(len(df3), 712)
        self.assertEqual(df3.columns[len(df3.columns)-1], 'y_pred')
        self.assertEqual(res2['response']['n_rows'], 0)

    def test_lgbm_classifier(self):
        df = load_titanic_dataset()
        res = analyzer.propensity.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Survived', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='lgbm-classifier', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
        model_id = res['model_id']
        self.assertEqual(len(res['features']), 10)
        self.assertEqual(res['confusion_matrix'].shape, (2, 2))
        self.assertEqual(len(res['stats']), 7)
        res = analyzer.propensity.predict(df, model_id=model_id, client_id=CLIENT_ID,
            idx_var='PassengerId', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            buffer_batch_size=1000, verbose=VERBOSE)
        res2 = analyzer.propensity.buffer_usage(client_id=CLIENT_ID)
        df3 = res['data2']
        self.assertEqual(len(df3), 712)
        self.assertEqual(df3.columns[len(df3.columns)-1], 'y_pred')
        self.assertEqual(res2['response']['n_rows'], 0)

class RegressionTest(unittest.TestCase):

    def test_lasso_regression(self):
        df = load_titanic_dataset()
        res = analyzer.regression.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Age', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            algorithm='lasso-regression', train_size=0.7, buffer_batch_size=1000, verbose=VERBOSE)
        model_id = res['model_id']
        self.assertEqual(len(res['features']), 10)
        self.assertEqual(len(res['stats']), 6)
        self.assertEqual(len(res['coefs']), 10)
        res = analyzer.regression.predict(df, model_id=model_id, client_id=CLIENT_ID,
            idx_var='PassengerId', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            buffer_batch_size=1000, verbose=VERBOSE)
        res2 = analyzer.regression.buffer_usage(client_id=CLIENT_ID)
        df3 = res['data2']
        self.assertEqual(len(df3), 712)
        self.assertEqual(df3.columns[len(df3.columns)-1], 'y_pred')
        self.assertEqual(res2['response']['n_rows'], 0)

class CausalTest(unittest.TestCase):

    def test_propensity_score_matching_ci(self):
        df = load_causal_dataset_v5()
        res = analyzer.causal.train(df, client_id=CLIENT_ID,
            idx_var='RecordId', outcome_var='Outcome', treatment_var='Treatment', categorical_vars=[], numerical_vars=['w0', 'w1', 's'],
            algorithm='propensity-score-matching-ci', 
            buffer_batch_size=1000, verbose=VERBOSE, encoding=True)
        model_id = res['model_id']
        self.assertEqual(len(res['atx']), 9) 
        self.assertEqual(len(res['raw']), 4)
        self.assertEqual(len(res['misc']), 10)
        self.assertEqual(len(res['bins']), 10)
        self.assertTrue(abs(res['atx'].loc['1']['Value']-0.2)/0.2 <= EPSILON) # ATT = 0.2

class MMMTest(unittest.TestCase):

    def test_mmm_train_with_encoding(self):
        df = load_mmm_dataset().head(30)
        res = analyzer.mmm.train(df, client_id=CLIENT_ID, 
            idx_var=None, time_var='wk_strt_dt', outcome_var='sales', 
            media_vars=['direct_mail', 'insert', 'newspaper', 'radio', 'tv', 'social_media', 'online_display'], 
            other_vars=[], 
            algorithm='mmm-carryover', 
            buffer_batch_size=1000, verbose=VERBOSE, encoding=True)
        model_id = res['model_id']
        train_stats = res['train_stats']
        test_stats = res['test_stats']
        lag_stats = res['lag_stats']
        contrib_stats = res['contrib_stats']
        self.assertTrue( abs( (float(train_stats['Value'][1]) - 0.666750) / 0.666750 ) < EPSILON )
        self.assertTrue( abs( (float(test_stats['Value'][1]) - 0.368148) / 0.368148 ) < EPSILON )
        self.assertEqual(lag_stats.shape, (8, 8))
        self.assertEqual(contrib_stats.shape, (17, 9))

class PerformanceTest(unittest.TestCase):

    def test_analyze_train_with_encoding(self):
        df = load_performance_analysis_dataset()
        res = analyzer.performance.train(df, client_id=CLIENT_ID, 
            idx_var=None, time_var='period', outcome_var='net_additions', 
            primary_vars=['beginning_balance', 'gross_connects', 'disconnects_voluntary', 'disconnects_involuntary', 'disconnects_total', 'net_additions', 'ending_balance', 'churn_voluntary', 'churn_involuntary'], 
            dimensional_vars=['region', 'market', 'product'], 
            edges=[
                ('ending_balance', 'beginning_balance'), 
                ('ending_balance', 'net_additions'), 
                ('net_additions', 'gross_connects'), 
                ('net_additions', 'disconnects_total'), 
                ('disconnects_total', 'disconnects_voluntary'), 
                ('disconnects_total', 'disconnects_involuntary'), 
                ('disconnects_voluntary', 'beginning_balance'), 
                ('disconnects_voluntary', 'churn_voluntary'), 
                ('disconnects_involuntary', 'beginning_balance'), 
                ('disconnects_involuntary', 'churn_involuntary'), 
            ], 
            hierarchies=[
                {
                    'name': 'Region', 
                    'dimension': 'region', 
                    'child': {
                        'name': 'Market', 
                        'dimension': 'market', 
                        'child': None, 
                    }
                },  
                {
                    'name': 'Product', 
                    'dimension': 'product', 
                    'child': None, 
                },  
            ], 
            udf={
                'churn_voluntary': 'DIVIDE(disconnects_voluntary, beginning_balance)', 
                'churn_involuntary': 'DIVIDE(disconnects_involuntary, beginning_balance)', 
            }, 
            buffer_batch_size=1000, verbose=VERBOSE, encoding=True)
        model_id = res['model_id']
        analysis = res['analysis']
        self.assertEqual(len(analysis.keys()), 3)
        self.assertEqual(analysis['drivers']['measure'], 'net_additions')
        self.assertEqual(int(analysis['drivers']['stats']['current']), -2859)

class RunnerBaseTest(unittest.TestCase):

    def test_load_keys_and_encode(self):
        request_id = str(uuid.uuid4())
        categorical_vars = ['Sex', 'Embarked'] 
        numerical_vars = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']   
        idx_var = ['PassengerId']
        bool_vars = []
        df = load_titanic_dataset()
        keys = analyzer.cluster._keys_load(model_id=request_id, verbose=VERBOSE)
        data, xref, zref, rref, fref, fref_exp, bref = analyzer.cluster._encode(
            df, categorical_vars=categorical_vars, numerical_vars=numerical_vars,
            bool_vars=bool_vars, record_id_var=idx_var, verbose=VERBOSE, keys=keys)
        self.assertEqual(len(data), len(df))
        self.assertEqual(fref['forward']['Embarked'], 'X_8')
        self.assertEqual(fref['reverse']['X_8'], 'Embarked')
        self.assertTrue(abs(zref['Parch']['mean']/0.43258426966292135 - 1) < EPSILON)
        self.assertTrue(abs(zref['Pclass']['mean']/2.240168539325843 - 1) < EPSILON)
        self.assertTrue(abs(zref['SibSp']['stdev']/0.9306921267673428 - 1) < EPSILON)
        self.assertTrue(abs(zref['Fare']['mean']/34.567251404494385 - 1) < EPSILON)
        self.assertTrue(abs(zref['Fare']['stdev']/52.93864817471089 - 1) < EPSILON)

    def test_load_keys_and_decode(self):
        request_id = str(uuid.uuid4())
        categorical_vars = ['Sex', 'Embarked'] 
        numerical_vars = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']   
        idx_var = ['PassengerId']
        bool_vars = []
        df = load_titanic_dataset()
        keys = analyzer.cluster._keys_load(model_id=request_id, verbose=VERBOSE)
        data, xref, zref, rref, fref, fref_exp, bref = analyzer.cluster._encode(
            df, categorical_vars=categorical_vars, numerical_vars=numerical_vars,
            bool_vars=bool_vars, record_id_var=idx_var, verbose=VERBOSE, keys=keys)
        self.assertEqual(len(data), len(df))
        self.assertEqual(fref['forward']['Embarked'], 'X_8')
        self.assertEqual(fref['reverse']['X_8'], 'Embarked')
        self.assertTrue(abs(zref['Parch']['mean']/0.43258426966292135 - 1) < EPSILON)
        self.assertTrue(abs(zref['Pclass']['mean']/2.240168539325843 - 1) < EPSILON)
        self.assertTrue(abs(zref['SibSp']['stdev']/0.9306921267673428 - 1) < EPSILON)
        self.assertTrue(abs(zref['Fare']['mean']/34.567251404494385 - 1) < EPSILON)
        self.assertTrue(abs(zref['Fare']['stdev']/52.93864817471089 - 1) < EPSILON)
        df2 = analyzer.cluster._decode(
                data, categorical_vars=categorical_vars,
                numerical_vars=numerical_vars, record_id_var=idx_var[0],xref=xref,
                zref=zref, rref=rref, fref=fref, verbose=VERBOSE)
        self.assertEqual(len(df2), len(data))
        self.assertEqual(len(df2), len(df))
