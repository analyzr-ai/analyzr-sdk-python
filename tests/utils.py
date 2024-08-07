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
import pandas as pd

def load_titanic_dataset():
    """
    :return df:
    """
    df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/titanic.csv', encoding = "ISO-8859-1", low_memory=False)
    df = df[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].dropna()
    df['PassengerId'] = df['PassengerId'].astype('string')
    return df

def load_banking_dataset():
    """
    :return df:
    """
    df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/bank_full_with_index.csv', encoding = "ISO-8859-1", low_memory=False)
    df = df[['index', 'campaign', 'duration', 'balance', 'y', 'housing', 'education', 'marital', 'job', 'loan', 'default', 'age']].dropna()
    df['index'] = df['index'].astype('string')
    return df

def generate_synthetic_dataset(n_features=2, n_samples=1000):
    """
    :param n_features:
    :return df:
    :return vars:
    """
    df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/synthetic_data_2_1000.csv', encoding = "ISO-8859-1", low_memory=False)
    vars = df.columns[1:]
    return df, vars

def load_churn_dataset():
    """
    :return df:
    """
    df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/public_datasets/telco-churn-v2.csv', encoding = "ISO-8859-1", low_memory=False)
    df['treatment'] = df['OnlineBackup']=='Yes'
    df = df[['customerID', 'Churn', 'treatment', 'SeniorCitizen', 'tenure', 'MonthlyCharges']].dropna().reset_index(drop=True)
    df['customerID'] = df['customerID'].astype('string')
    return df

def load_causal_dataset_v5():
    """
    :return df:
    """
    df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/public_datasets/synthetic_dataset_nonlinear_refactored_v5_20230719D2.csv', encoding = "ISO-8859-1", low_memory=False)
    df = df[['RecordId', 'Outcome', 'Treatment', 'w0', 'w1', 's']].dropna().reset_index(drop=True)
    df['RecordId'] = df['RecordId'].astype('string')
    df['Treatment'] = df['Treatment'].astype('bool')
    return df

def load_performance_analysis_dataset():
    """
    :return df:
    """
    df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/public_datasets/subscriber_data_clean.csv', encoding = "ISO-8859-1", low_memory=False)
    return df 

def load_mmm_dataset():
    """
    :return df:
    """
    df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/public_datasets/mmm2.csv', encoding = "ISO-8859-1", low_memory=False)
    return df[['wk_strt_dt', 'sales', 'direct_mail', 'insert', 'newspaper', 'radio', 'tv', 'social_media', 'online_display']] 
