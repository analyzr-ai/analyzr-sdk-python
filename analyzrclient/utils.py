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
import uuid
import numpy as np
import pandas as pd
from copy import deepcopy

def xref_encode(series):
    """
    Replaces categorical values in a Series with UUIDs

    :param series:
    :return series2:
    :return xref:
    """
    # generate xref
    values = series.value_counts().index
    xref = {'forward': {}, 'reverse': {}}
    for val in values:
        if val is not None and val is not np.nan: 
            key = str(uuid.uuid4())
            xref['forward'][str(val)] = key
            xref['reverse'][key] = str(val)

    # convert series
    series2, xref = xref_encode_with_keys(series, xref)
    return series2, xref

def xref_encode_with_keys(series, xref):
    """
    Replaces categorical values in a Series with UUIDs using existing encoding keys

    :param series:
    :param xref:
    :return series2:
    :return xref:
    """
    series2 = deepcopy(series)
    skipped_vals = []
    for idx, val in series2.items():
        if val is not None and val is not np.nan:
            if str(val) not in xref['forward'].keys():
                # print('Value is not in xref forward', str(val))
                # Adding missing value to keys
                key = str(uuid.uuid4())
                xref['forward'][str(val)] = key
                xref['reverse'][key] = str(val)
                if str(val) not in skipped_vals: skipped_vals.append(str(val))
                # print('Value is in xref forward', str(val))
            series2[idx] = xref['forward'][str(val)]
        else:
            series2[idx] = None
    if len(skipped_vals)>0:
        print('        WARNING! The following values were not present in the training encoding set and will be skipped: ', skipped_vals)
    return series2, xref

def xref_decode(series, xref, verbose=False):
    """
    Replaces UUIDs in a Series with matching categorical values

    :param series:
    :param xref:
    :return series2:
    """
    series2 = deepcopy(series)
    for idx, val in series.items():
        series2[idx] = xref['reverse'][str(val)]
    return series2

def zref_encode(series):
    """
    Replaces numerical values in a Series with z scores

    :param series:
    :return series2:
    :return zref:
    """
    zref = { 'mean': series.mean(), 'stdev': series.std() }
    series2 = zref_encode_with_keys(series, zref)
    return series2, zref

def zref_encode_with_keys(series, zref):
    """
    Replaces numerical values in a Series with z scores using existing encoding keys

    :param series:
    :param zref:
    :return series2:
    """
    series2 = deepcopy(series)

    # Remove mean
    if np.isnan(zref['mean'])==False:
        series2 -= zref['mean']
    else:
        print('ERROR! series mean is nan, cannot encode series')
        return series, {}

    # Scale with std deviation
    if np.isnan(zref['stdev'])==False and zref['stdev']!=0.0:
        series2 /= zref['stdev']

    return series2

def zref_decode(series, zref):
    """
    Replaces z scores in a Series with denormalized numerical values

    :param series:
    :param zref:
    :return series2:
    """
    series2 = deepcopy(series)

    # Rescale with std deviation
    if np.isnan(zref['stdev'])==False and zref['stdev']!=0.0:
        series2 *= zref['stdev']

    # Add back mean
    if np.isnan(zref['mean'])==False:
        series2 += zref['mean']

    return series2

def zref_decode_first_derivative_value(val, zref_x, zref_y):
    """
    Replaces z scores in a Series with denormalized numerical values. 
    Applies to first derivative of variable y with respect to variable 
    x, such as coefficients for linear regression. 

    :param val:
    :param zref_x:
    :param zref_y:
    :return series2:
    """
    new_val = val
    if np.isnan(zref_y['stdev'])==False and zref_y['stdev']>0.0 and np.isnan(zref_x['stdev'])==False and zref_x['stdev']>0.0:
        new_val *= (zref_y['stdev'] / zref_x['stdev'])
    return new_val

def bref_encode(series):
    """
    Boolean variables are not encoded at this stage

    :param series:
    :return series2:
    :return bref:
    """
    bref = {}
    series2 = bref_encode_with_keys(series, bref)
    return series2, bref

def bref_encode_with_keys(series, bref):
    """
    Boolean variables are not encoded at this stage

    :param series:
    :return series2:
    :return bref:
    """
    series2 = deepcopy(series)
    return series2

def bref_decode(series, bref):
    """
    Boolean variables are not encoded at this stage

    :param series:
    :param bref:
    :return series2:
    """
    series2 = deepcopy(series)
    return series2

def rref_encode(df, record_id_var):
    """
    Encode record ID column

    :param df:
    :param record_id_var:
    :return df2:
    :return rref:
    """
    if isinstance(df, pd.DataFrame)==False:
        print('rref_encode() requires a pandas DataFrame object as input', type=type(df))
        return None, None
    df2 = deepcopy(df)
    df2[record_id_var], rref = xref_encode(df[record_id_var])
    return df2, rref

def rref_encode_with_keys(df, record_id_var, rref):
    """
    Encode record ID column using existing encoding keys

    :param df:
    :param record_id_var:
    :param rref:
    :return df2:
    """
    if isinstance(df, pd.DataFrame)==False:
        print('rref_encode() requires a pandas DataFrame object as input', type=type(df))
        return None, None
    df2 = deepcopy(df)
    df2[record_id_var], rref = xref_encode_with_keys(df[record_id_var], rref)
    return df2, rref

def rref_decode(df, record_id_var, rref, verbose=False):
    """
    :param df:
    :param record_id_var:
    :param rref:
    :return df2:
    """
    if rref=={}: return df
    df2 = deepcopy(df)
    if record_id_var not in df2.columns:
        df2[record_id_var] = df2.index
    series = df[record_id_var].squeeze() if isinstance(df[record_id_var], pd.DataFrame) else df[record_id_var]
    df2[record_id_var] = xref_decode(series, rref, True)
    df2 = df2.set_index(record_id_var)
    return df2

def fref_encode(df):
    """
    Encode field names

    :param df:
    :return df2:
    :return fref:
    """
    df2 = deepcopy(df)
    fref = {'forward': {'PC_ID': 'PC_ID'}, 'reverse': {'PC_ID': 'PC_ID'}}
    counter = 0
    cols2 = []
    for col in df2.columns:
        key = 'X_{}'.format(counter)
        fref['forward'][col] = key
        fref['reverse'][key] = col
        cols2.append(key)
        counter += 1
    df2.columns = cols2
    return df2, fref

def fref_encode_with_keys(df, fref):
    """
    Encode field names using existing encoding keys

    :param df:
    :param fref:
    :return df2:
    """
    df2 = deepcopy(df)
    df2.columns = [ fref['forward'][col] for col in df2.columns ]
    return df2

def fref_to_fref_expanded(fref, xref):
    """
    Convert <fref> to expanded field names (using dummy variable convention)

    :param fref:
    :param xref:
    :return fref_exp:
    """
    fref_exp = {'forward': {}, 'reverse': {}}
    for col in fref['forward'].keys():
        key = fref['forward'][col]
        if col in xref.keys():
            # col is categorical, need to expand it
            for category in xref[col]['forward'].keys():
                col_exp = '{}_{}'.format(col, category)
                key_exp = '{}_{}'.format(key, xref[col]['forward'][category])
                fref_exp['forward'][col_exp] = key_exp
                fref_exp['reverse'][key_exp] = col_exp
        else:
            # col is not categorical
            fref_exp['forward'][col] = key
            fref_exp['reverse'][key] = col
    return fref_exp

def fref_decode(df, fref):
    """
    Decode field names

    :param df:
    :param fref:
    :return df2:
    """
    if fref=={}: return df
    df2 = deepcopy(df)
    # df2.columns = [ fref['reverse'][col] for col in df2.columns ]
    df2.columns = fref_decode_columns(df2.columns, fref)
    return df2

def fref_decode_columns(cols, fref):
    """
    Decode array of column names

    :param cols:
    :param fref:
    :return cols2:
    """
    if fref=={}: return cols
    cols2 = [ fref_decode_value(col, fref) for col in cols ]
    return cols2

def fref_decode_value(col, fref):
    """
    Decode single column name

    :param col:
    :param fref:
    :return col2:
    """
    col2 = col
    if fref!={} and col in fref['reverse'].keys():
        col2 =  fref['reverse'][col]
    return col2

def compute_cluster_stats(df, categorical_fields):
    """
    :param df:
    :param categorical_fields:
    :return stats:
    """
    if df is None: return None
    stats = pd.DataFrame(df['PC_ID'].value_counts())
    stats.columns=['count']
    stats['frequency'] = stats['count']/len(df)
    stats.sort_index(inplace=True)
    stats.index.rename('PC_ID', inplace=True)
    stats = stats.join(pd.get_dummies(df, columns=categorical_fields).groupby('PC_ID').mean(numeric_only=True), on='PC_ID')
    return stats.T

def compute_cluster_distances(df, categorical_fields):
    """
    :param df:
    :param categorical_fields:
    :return distances:
    """
    if df is None: return None
    df2 = pd.get_dummies(df, columns=categorical_fields).groupby('PC_ID').mean(numeric_only=True)
    recs = []
    for idx1, cluster1 in df2.iterrows():
        rec = []
        for idx2, cluster2 in df2.iterrows():
            rec.append(np.linalg.norm(cluster1-cluster2))
        recs.append(rec)
    return pd.DataFrame.from_records(recs)

def merge_cluster_ids(df, pc_id, idx_var):
    """
    Merge principal component IDs back onto original records and compute stats by cluster

    :param df: original dataframe to be clustered
    :param pc_id: principal component IDs by record
    :param idx_var: name of record ID field
    :return df2:
    """
    pc_id[idx_var] = pc_id[idx_var].astype(df[idx_var].dtype)
    return pd.merge(df, pc_id, left_on=idx_var, right_on=idx_var, how='left')

def get_test_data():
    """
    :param:
    :return df:
    """
    df = pd.read_csv('https://g2mstaticfiles.blob.core.windows.net/$web/titanic.csv', encoding = "ISO-8859-1", low_memory=False)
    return df
