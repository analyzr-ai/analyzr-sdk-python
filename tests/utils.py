"""
Copyright (c) 2020-2021 Go2Market Insights, LLC
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

def generate_synthetic_dataset(n_features=2, n_samples=1000):
    """
    :param n_features:
    :return df:
    :return vars:
    """
    df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/synthetic_data_2_1000.csv', encoding = "ISO-8859-1", low_memory=False)
    vars = df.columns[1:]
    return df, vars
