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

    def test_pca_kmeans(self):
        df = load_titanic_dataset()
        res = analyzer.cluster.run(df, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='pca-kmeans', buffer_batch_size=1000, verbose=VERBOSE)
        df2 = res['data']
        model_id = res['request_id']
        res = analyzer.cluster.buffer_usage(client_id=CLIENT_ID)
        self.assertEqual(df2.iloc[0]['Fare'], 7.25)
        # self.assertEqual(df2.iloc[0]['PC_ID'], 2)
        self.assertEqual(df2.iloc[711]['Embarked'], 'Q')
        # self.assertEqual(df2.iloc[711]['PC_ID'], 4)
        self.assertEqual(res['response']['n_rows'], 0)
        res = analyzer.cluster.predict(df, model_id=model_id, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            buffer_batch_size=1000, verbose=VERBOSE)
        df3 = res['data2']
        self.assertEqual(len(df3), 712)
        self.assertEqual(df3.columns[0], 'PC_ID')

    def test_pca_kmeans_simple(self):
        df = load_titanic_dataset()
        res = analyzer.cluster.run(df, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='pca-kmeans-simple', buffer_batch_size=1000, verbose=VERBOSE)
        df2 = res['data']
        model_id = res['request_id']
        res = analyzer.cluster.buffer_usage(client_id=CLIENT_ID)
        self.assertEqual(df2.iloc[0]['Fare'], 7.25)
        # self.assertEqual(df2.iloc[0]['PC_ID'], 2)
        self.assertEqual(df2.iloc[711]['Embarked'], 'Q')
        # self.assertEqual(df2.iloc[711]['PC_ID'], 4)
        self.assertEqual(res['response']['n_rows'], 0)
        res = analyzer.cluster.predict(df, model_id=model_id, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            buffer_batch_size=1000, verbose=VERBOSE)
        df3 = res['data2']
        self.assertEqual(len(df3), 712)
        self.assertEqual(df3.columns[0], 'PC_ID')

    def test_inc_pca_kmeans(self):
        df = load_titanic_dataset()
        res = analyzer.cluster.run(df, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='incremental-pca-kmeans', buffer_batch_size=1000, cluster_batch_size=150, verbose=VERBOSE)
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

    def test_kmeans(self):
        df = load_titanic_dataset()
        res = analyzer.cluster.run(df, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='kmeans', buffer_batch_size=1000, cluster_batch_size=150, verbose=VERBOSE)
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

    def test_minibatchkmeans(self):
        df = load_titanic_dataset()
        res = analyzer.cluster.run(df, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='minibatch-kmeans', buffer_batch_size=1000, cluster_batch_size=150, verbose=VERBOSE)
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

    def test_gaussianmixture(self):
        df = load_titanic_dataset()
        res = analyzer.cluster.run(df, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='gaussian-mixture', buffer_batch_size=1000, cluster_batch_size=150, verbose=VERBOSE)
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

    def test_dbscan(self):
        df = load_titanic_dataset()
        res = analyzer.cluster.run(df, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='dbscan', buffer_batch_size=1000, cluster_batch_size=150, verbose=VERBOSE)
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

    def test_optics(self):
        df = load_titanic_dataset()
        res = analyzer.cluster.run(df, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='optics', buffer_batch_size=1000, cluster_batch_size=150, verbose=VERBOSE)
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

    def test_spectralclustering(self):
        df = load_titanic_dataset()
        res = analyzer.cluster.run(df, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='spectral-clustering', buffer_batch_size=1000, cluster_batch_size=150, verbose=VERBOSE)
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

    def test_hierarchicalagglomerative(self):
        df = load_titanic_dataset()
        res = analyzer.cluster.run(df, client_id=CLIENT_ID, idx_var='PassengerId',
            categorical_vars=['Sex', 'Embarked'], numerical_vars=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='hierarchical-agglomerative', buffer_batch_size=1000, cluster_batch_size=150, verbose=VERBOSE)
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

    def test_random_forest_classifier(self):
        df = load_titanic_dataset()
        res = analyzer.propensity.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Survived', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='random-forest-classifier', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
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

    def test_gradient_boosting_classifier(self):
        df = load_titanic_dataset()
        res = analyzer.propensity.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Survived', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='gradient-boosting-classifier', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
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

    def test_xgboost_classifier(self):
        df = load_titanic_dataset()
        res = analyzer.propensity.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Survived', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='xgboost-classifier', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
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

    def test_extra_trees_classifier(self):
        df = load_titanic_dataset()
        res = analyzer.propensity.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Survived', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='extra-trees-classifier', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
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

    def test_h2o_random_forest_classifier(self):
        df = load_titanic_dataset()
        res = analyzer.propensity.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Survived', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='h2o-random-forest-classifier', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
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

    def test_h2o_gradient_boosting_classifier(self):
        df = load_titanic_dataset()
        res = analyzer.propensity.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Survived', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='h2o-gradient-boosting-classifier', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
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

    def test_h2o_xgboost_classifier(self):
        df = load_titanic_dataset()
        res = analyzer.propensity.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Survived', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
            algorithm='h2o-xgboost-classifier', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
        model_id = res['model_id']
        self.assertTrue(len(res['features'])==9 or len(res['features'])==10) # the number of features tends to vary between 9 and 10 depending on the training sample
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

    def test_random_forest_regression(self):
        df = load_titanic_dataset()
        res = analyzer.regression.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Age', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            algorithm='random-forest-regression', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
        model_id = res['model_id']
        self.assertEqual(len(res['features']), 10)
        self.assertEqual(len(res['stats']), 6)
        res = analyzer.regression.predict(df, model_id=model_id, client_id=CLIENT_ID,
            idx_var='PassengerId', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            buffer_batch_size=1000, verbose=VERBOSE)
        res2 = analyzer.regression.buffer_usage(client_id=CLIENT_ID)
        df3 = res['data2']
        self.assertEqual(len(df3), 712)
        self.assertEqual(df3.columns[len(df3.columns)-1], 'y_pred')
        self.assertEqual(res2['response']['n_rows'], 0)

    def test_gradient_boosting_regression(self):
        df = load_titanic_dataset()
        res = analyzer.regression.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Age', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            algorithm='gradient-boosting-regression', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
        model_id = res['model_id']
        self.assertEqual(len(res['features']), 10)
        self.assertEqual(len(res['stats']), 6)
        res = analyzer.regression.predict(df, model_id=model_id, client_id=CLIENT_ID,
            idx_var='PassengerId', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            buffer_batch_size=1000, verbose=VERBOSE)
        res2 = analyzer.regression.buffer_usage(client_id=CLIENT_ID)
        df3 = res['data2']
        self.assertEqual(len(df3), 712)
        self.assertEqual(df3.columns[len(df3.columns)-1], 'y_pred')
        self.assertEqual(res2['response']['n_rows'], 0)

    def test_xgboost_regression(self):
        df = load_titanic_dataset()
        res = analyzer.regression.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Age', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            algorithm='xgboost-regression', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
        model_id = res['model_id']
        self.assertEqual(len(res['features']), 10)
        self.assertEqual(len(res['stats']), 6)
        res = analyzer.regression.predict(df, model_id=model_id, client_id=CLIENT_ID,
            idx_var='PassengerId', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            buffer_batch_size=1000, verbose=VERBOSE)
        res2 = analyzer.regression.buffer_usage(client_id=CLIENT_ID)
        df3 = res['data2']
        self.assertEqual(len(df3), 712)
        self.assertEqual(df3.columns[len(df3.columns)-1], 'y_pred')
        self.assertEqual(res2['response']['n_rows'], 0)

    def test_linear_regression(self):
        df = load_titanic_dataset()
        res = analyzer.regression.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Age', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            algorithm='linear-regression', train_size=0.5, buffer_batch_size=1000, verbose=VERBOSE)
        model_id = res['model_id']
        self.assertEqual(len(res['features']), 10)
        self.assertEqual(len(res['stats']), 6)
        res = analyzer.regression.predict(df, model_id=model_id, client_id=CLIENT_ID,
            idx_var='PassengerId', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            buffer_batch_size=1000, verbose=VERBOSE)
        res2 = analyzer.regression.buffer_usage(client_id=CLIENT_ID)
        df3 = res['data2']
        self.assertEqual(len(df3), 712)
        self.assertEqual(df3.columns[len(df3.columns)-1], 'y_pred')
        self.assertEqual(res2['response']['n_rows'], 0)

    def test_lasso_regression(self):
        df = load_titanic_dataset()
        res = analyzer.regression.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Age', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            algorithm='lasso-regression', train_size=0.7, buffer_batch_size=1000, verbose=VERBOSE)
        model_id = res['model_id']
        self.assertEqual(len(res['features']), 10)
        self.assertEqual(len(res['stats']), 6)
        res = analyzer.regression.predict(df, model_id=model_id, client_id=CLIENT_ID,
            idx_var='PassengerId', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            buffer_batch_size=1000, verbose=VERBOSE)
        res2 = analyzer.regression.buffer_usage(client_id=CLIENT_ID)
        df3 = res['data2']
        self.assertEqual(len(df3), 712)
        self.assertEqual(df3.columns[len(df3.columns)-1], 'y_pred')
        self.assertEqual(res2['response']['n_rows'], 0)

    def test_ridge_regression(self):
        df = load_titanic_dataset()
        res = analyzer.regression.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Age', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            algorithm='ridge-regression', train_size=0.7, buffer_batch_size=1000, verbose=VERBOSE)
        model_id = res['model_id']
        self.assertEqual(len(res['features']), 10)
        self.assertEqual(len(res['stats']), 6)
        res = analyzer.regression.predict(df, model_id=model_id, client_id=CLIENT_ID,
            idx_var='PassengerId', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            buffer_batch_size=1000, verbose=VERBOSE)
        res2 = analyzer.regression.buffer_usage(client_id=CLIENT_ID)
        df3 = res['data2']
        self.assertEqual(len(df3), 712)
        self.assertEqual(df3.columns[len(df3.columns)-1], 'y_pred')
        self.assertEqual(res2['response']['n_rows'], 0)

    def test_bayesian_ridge_regression(self):
        df = load_titanic_dataset()
        res = analyzer.regression.train(df, client_id=CLIENT_ID,
            idx_var='PassengerId', outcome_var='Age', categorical_vars=['Sex', 'Embarked'], numerical_vars=['Pclass', 'Survived', 'SibSp', 'Parch', 'Fare', 'Age'],
            algorithm='bayesian-ridge-regression', train_size=0.7, buffer_batch_size=1000, verbose=VERBOSE)
        model_id = res['model_id']
        self.assertEqual(len(res['features']), 10)
        self.assertEqual(len(res['stats']), 6)
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

    def test_propensity_score_blocking_ci(self):
        df = load_causal_dataset_v5()
        res = analyzer.causal.train(df, client_id=CLIENT_ID,
            idx_var='RecordId', outcome_var='Outcome', treatment_var='Treatment', categorical_vars=[], numerical_vars=['w0', 'w1', 's'],
            algorithm='propensity-score-blocking-ci', 
            buffer_batch_size=1000, verbose=VERBOSE, encoding=True)
        model_id = res['model_id']
        self.assertEqual(len(res['atx']), 9) 
        self.assertEqual(len(res['raw']), 4)
        self.assertEqual(len(res['misc']), 10)
        self.assertEqual(len(res['bins']), 0)
        self.assertTrue(abs(res['atx'].loc['1']['Value']-0.234055)/0.234055 <= EPSILON*10) # ATT = 0.234055

    def test_propensity_score_weighting_ci(self):
        df = load_causal_dataset_v5()
        res = analyzer.causal.train(df, client_id=CLIENT_ID,
            idx_var='RecordId', outcome_var='Outcome', treatment_var='Treatment', categorical_vars=[], numerical_vars=['w0', 'w1', 's'],
            algorithm='propensity-score-weighting-ci', 
            buffer_batch_size=1000, verbose=VERBOSE, encoding=True)
        model_id = res['model_id']
        self.assertEqual(len(res['atx']), 9) 
        self.assertEqual(len(res['raw']), 4)
        self.assertEqual(len(res['misc']), 10)
        self.assertEqual(len(res['bins']), 0)
        self.assertTrue(abs(res['atx'].loc['2']['Value']-0.180893)/0.180893 <= EPSILON) # ATE = 0.180893

    def test_ols_ci(self):
        df = load_causal_dataset_v5()
        res = analyzer.causal.train(df, client_id=CLIENT_ID,
            idx_var='RecordId', outcome_var='Outcome', treatment_var='Treatment', categorical_vars=[], numerical_vars=['w0', 'w1', 's'],
            algorithm='ols-ci', 
            buffer_batch_size=1000, verbose=VERBOSE, encoding=True)
        model_id = res['model_id']
        self.assertEqual(len(res['atx']), 9) 
        self.assertEqual(len(res['raw']), 4)
        self.assertEqual(len(res['misc']), 10)
        self.assertEqual(len(res['bins']), 0)
        self.assertTrue(abs(res['atx'].loc['1']['Value']-0.270463)/0.270463 <= EPSILON*10) # ATT = 0.270463

    def test_propensity_score_matching_dw(self):
        df = load_causal_dataset_v5()
        res = analyzer.causal.train(df, client_id=CLIENT_ID,
            idx_var='RecordId', outcome_var='Outcome', treatment_var='Treatment', categorical_vars=[], numerical_vars=['w0', 'w1', 's'],
            algorithm='propensity-score-matching-dw', standard_error=False, 
            buffer_batch_size=1000, verbose=VERBOSE, encoding=True)
        model_id = res['model_id']
        self.assertEqual(len(res['atx']), 9) 
        self.assertEqual(len(res['raw']), 4)
        self.assertEqual(len(res['misc']), 10)
        self.assertEqual(len(res['bins']), 0)
        self.assertTrue(abs(res['atx'].loc['1']['Value']-0.2)/0.2 <= EPSILON) # ATT = 0.2

    def test_propensity_score_weighting_dw(self):
        df = load_causal_dataset_v5()
        res = analyzer.causal.train(df, client_id=CLIENT_ID,
            idx_var='RecordId', outcome_var='Outcome', treatment_var='Treatment', categorical_vars=[], numerical_vars=['w0', 'w1', 's'],
            algorithm='propensity-score-weighting-dw', standard_error=False, 
            buffer_batch_size=1000, verbose=VERBOSE, encoding=True)
        model_id = res['model_id']
        self.assertEqual(len(res['atx']), 9) 
        self.assertEqual(len(res['raw']), 4)
        self.assertEqual(len(res['misc']), 10)
        self.assertEqual(len(res['bins']), 0)
        self.assertTrue(abs(res['atx'].loc['1']['Value']-0.710537)/0.710537 <= EPSILON*10) # ATT = 0.710537

    def test_propensity_score_stratification_dw(self):
        df = load_causal_dataset_v5()
        res = analyzer.causal.train(df, client_id=CLIENT_ID,
            idx_var='RecordId', outcome_var='Outcome', treatment_var='Treatment', categorical_vars=[], numerical_vars=['w0', 'w1', 's'],
            algorithm='propensity-score-stratification-dw', standard_error=False, 
            buffer_batch_size=1000, verbose=VERBOSE, encoding=True)
        model_id = res['model_id']
        self.assertEqual(len(res['atx']), 9) 
        self.assertEqual(len(res['raw']), 4)
        self.assertEqual(len(res['misc']), 10)
        self.assertEqual(len(res['bins']), 0)
        self.assertTrue(abs(res['atx'].loc['1']['Value']-0.212232)/0.212232 <= EPSILON*10) # ATT = 0.212232

class MMMTest(unittest.TestCase):

    def test_mmm_train_no_encoding(self):
        df = load_mmm_dataset().head(30)
        res = analyzer.mmm.train(df, client_id=CLIENT_ID, 
            idx_var=None, time_var='wk_strt_dt', outcome_var='sales', 
            media_vars=['direct_mail', 'insert', 'newspaper', 'radio', 'tv', 'social_media', 'online_display'], 
            other_vars=[], 
            algorithm='mmm-carryover', 
            buffer_batch_size=1000, verbose=VERBOSE, encoding=False)
        model_id = res['model_id']
        train_stats = res['train_stats']
        test_stats = res['test_stats']
        lag_stats = res['lag_stats']
        contrib_stats = res['contrib_stats']
        self.assertTrue( abs( (float(train_stats['Value'][1]) - 0.666750) / 0.666750 ) < EPSILON )
        self.assertTrue( abs( (float(test_stats['Value'][1]) - 0.368148) / 0.368148 ) < EPSILON )
        self.assertEqual(lag_stats.shape, (8, 8))
        self.assertEqual(contrib_stats.shape, (17, 9))

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

    def test_mmm_optimize_no_encoding(self):
        # Train
        df = load_mmm_dataset().head(30)
        res = analyzer.mmm.train(df, client_id=CLIENT_ID, 
            idx_var=None, time_var='wk_strt_dt', outcome_var='sales', 
            media_vars=['direct_mail', 'insert', 'newspaper', 'radio', 'tv', 'social_media', 'online_display'], 
            other_vars=[], 
            algorithm='mmm-adstock', 
            buffer_batch_size=1000, verbose=VERBOSE, encoding=False)
        model_id = res['model_id']
        # Optimize
        obj = analyzer.mmm.optimize(
            model_id=model_id, 
            client_id=CLIENT_ID, 
            budget=30e6, 
            encoding=False, 
        )
        data2 = obj['data2']
        self.assertEqual(data2.shape, (11, 2))
        self.assertEqual(float(data2['Value'][0]), 0)
        self.assertEqual(float(data2['Value'][1]), 48)
        self.assertEqual(float(data2['Value'][3]), 706)

    def test_mmm_optimize_with_encoding(self):
        # Train
        df = load_mmm_dataset().head(30)
        res = analyzer.mmm.train(df, client_id=CLIENT_ID, 
            idx_var=None, time_var='wk_strt_dt', outcome_var='sales', 
            media_vars=['direct_mail', 'insert', 'newspaper', 'radio', 'tv', 'social_media', 'online_display'], 
            other_vars=[], 
            algorithm='mmm-adstock', 
            buffer_batch_size=1000, verbose=VERBOSE, encoding=True)
        model_id = res['model_id']
        # Optimize
        obj = analyzer.mmm.optimize(
            model_id=model_id, 
            client_id=CLIENT_ID, 
            budget=30e6, 
            encoding=True, 
        )
        data2 = obj['data2']
        self.assertEqual(data2.shape, (11, 2))
        self.assertEqual(float(data2['Value'][0]), 0)
        self.assertEqual(float(data2['Value'][1]), 48)
        self.assertEqual(float(data2['Value'][3]), 706)


class PerformanceTest(unittest.TestCase):

    def test_analyze_train_no_encoding(self):
        df = load_performance_analysis_dataset()
        # print('\n[test_analyze_train] df: \n', df)
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
            buffer_batch_size=1000, verbose=VERBOSE, encoding=False)
        model_id = res['model_id']
        analysis = res['analysis']
        self.assertEqual(len(analysis.keys()), 3)
        self.assertEqual(analysis['drivers']['measure'], 'net_additions')
        self.assertEqual(int(analysis['drivers']['stats']['current']), -2859)

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

    def test_analyze_run_no_refresh_no_encoding(self):
        df = load_performance_analysis_dataset()
        # print('\n[test_analyze_train] df: \n', df)
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
            buffer_batch_size=1000, verbose=VERBOSE, encoding=False)
        model_id = res['model_id']
        analysis = res['analysis']
        self.assertEqual(len(analysis.keys()), 3)
        self.assertEqual(analysis['drivers']['measure'], 'net_additions')
        self.assertEqual(int(analysis['drivers']['stats']['current']), -2859)
        res = analyzer.performance.run(
            model_id=model_id, 
            client_id=CLIENT_ID, 
            time_var='period',
            outcome_var='net_additions',
            address=('Valecross', None, 'Subs'), 
            primary_vars=['beginning_balance', 'gross_connects', 'disconnects_voluntary', 'disconnects_involuntary', 'disconnects_total', 'net_additions', 'ending_balance', 'churn_voluntary', 'churn_involuntary'], 
            dimensional_vars=['region', 'market', 'product'], 
            encoding=False, 
        )
        analysis = res['analysis']
        self.assertEqual(len(analysis.keys()), 3)
        self.assertEqual(analysis['drivers']['measure'], 'net_additions')
        self.assertEqual(int(analysis['drivers']['stats']['current']), -883)

    def test_analyze_run_no_refresh_with_encoding(self):
        df = load_performance_analysis_dataset()
        # print('\n[test_analyze_train] df: \n', df)
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
        res = analyzer.performance.run(
            model_id=model_id, 
            client_id=CLIENT_ID, 
            time_var='period',
            outcome_var='net_additions',
            address=('Valecross', None, 'Subs'), 
            primary_vars=['beginning_balance', 'gross_connects', 'disconnects_voluntary', 'disconnects_involuntary', 'disconnects_total', 'net_additions', 'ending_balance', 'churn_voluntary', 'churn_involuntary'], 
            dimensional_vars=['region', 'market', 'product'], 
            encoding=True, 
        )
        analysis = res['analysis']
        self.assertEqual(len(analysis.keys()), 3)
        self.assertEqual(analysis['drivers']['measure'], 'net_additions')
        self.assertEqual(int(analysis['drivers']['stats']['current']), -883)

    def test_analyze_run_purge_no_reload_with_encoding(self):
        df = load_performance_analysis_dataset()
        # print('\n[test_analyze_train] df: \n', df)
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
        res = analyzer.performance.purge(
            model_id=model_id, 
            client_id=CLIENT_ID, 
            verbose=VERBOSE, 
        )
        self.assertEqual(res['status'], 'Complete')
        res = analyzer.performance.run(
            model_id=model_id, 
            client_id=CLIENT_ID, 
            time_var='period',
            outcome_var='net_additions',
            address=('Valecross', None, 'Subs'), 
            primary_vars=['beginning_balance', 'gross_connects', 'disconnects_voluntary', 'disconnects_involuntary', 'disconnects_total', 'net_additions', 'ending_balance', 'churn_voluntary', 'churn_involuntary'], 
            dimensional_vars=['region', 'market', 'product'], 
            encoding=True, 
        )
        analysis = res['analysis']
        self.assertEqual(len(analysis.keys()), 3)
        self.assertEqual(analysis['drivers'], {})

    def test_analyze_run_purge_and_reload_with_encoding(self):
        df = load_performance_analysis_dataset()
        # print('\n[test_analyze_train] df: \n', df)
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
        res = analyzer.performance.purge(
            model_id=model_id, 
            client_id=CLIENT_ID, 
            verbose=VERBOSE, 
        )
        self.assertEqual(res['status'], 'Complete')
        res = analyzer.performance.run(
            df=df, 
            model_id=model_id, 
            client_id=CLIENT_ID, 
            time_var='period',
            outcome_var='net_additions',
            address=('Valecross', None, 'Subs'), 
            primary_vars=['beginning_balance', 'gross_connects', 'disconnects_voluntary', 'disconnects_involuntary', 'disconnects_total', 'net_additions', 'ending_balance', 'churn_voluntary', 'churn_involuntary'], 
            dimensional_vars=['region', 'market', 'product'], 
            encoding=True, 
        )
        analysis = res['analysis']
        self.assertEqual(len(analysis.keys()), 3)
        self.assertEqual(analysis['drivers']['measure'], 'net_additions')
        self.assertEqual(int(analysis['drivers']['stats']['current']), -883)

    def test_analyze_run_purge_and_reload_no_encoding(self):
        df = load_performance_analysis_dataset()
        # print('\n[test_analyze_train] df: \n', df)
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
            buffer_batch_size=1000, verbose=VERBOSE, encoding=False)
        model_id = res['model_id']
        analysis = res['analysis']
        self.assertEqual(len(analysis.keys()), 3)
        self.assertEqual(analysis['drivers']['measure'], 'net_additions')
        self.assertEqual(int(analysis['drivers']['stats']['current']), -2859)
        res = analyzer.performance.purge(
            model_id=model_id, 
            client_id=CLIENT_ID, 
            verbose=VERBOSE, 
        )
        self.assertEqual(res['status'], 'Complete')
        res = analyzer.performance.run(
            df=df, 
            model_id=model_id, 
            client_id=CLIENT_ID, 
            time_var='period',
            outcome_var='net_additions',
            address=('Valecross', None, 'Subs'), 
            primary_vars=['beginning_balance', 'gross_connects', 'disconnects_voluntary', 'disconnects_involuntary', 'disconnects_total', 'net_additions', 'ending_balance', 'churn_voluntary', 'churn_involuntary'], 
            dimensional_vars=['region', 'market', 'product'], 
            encoding=False, 
        )
        analysis = res['analysis']
        self.assertEqual(len(analysis.keys()), 3)
        self.assertEqual(analysis['drivers']['measure'], 'net_additions')
        self.assertEqual(int(analysis['drivers']['stats']['current']), -883)

