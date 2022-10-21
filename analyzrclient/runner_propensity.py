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
import os, sys, time, json
import pandas as pd
from copy import deepcopy

from .runner_base import BaseRunner
from .constants import *
from .utils import *

class PropensityRunner(BaseRunner):
    """
    Runs the propensity scoring pipeline

    """
    def __init__(self, client=None, base_url=None):
        """
        """
        super().__init__(client=client, base_url=base_url)
        self.__uri = '{}/analytics/'.format(self._base_url)
        return

    def predict(self, df, model_id=None, client_id=None,
        idx_var=None, categorical_vars=[], numerical_vars=[], bool_vars=[],
        buffer_batch_size=1000, api_batch_size=2000, verbose=False, timeout=600, step=2,
        compressed=False, staging=True, encoding=True):
        """
        :param df:
        :param model_id:
        :param request_id:
        :param client_id:
        :param idx_var:
        :param categorical_vars:
        :param numerical_vars:
        :param buffer_batch_size:
        :param verbose:
        :param compressed:
        :param staging:
        :param encoding:
        :return res5:
        """
        res = {}
        res['model_id'] = model_id
        res['data2'] = pd.DataFrame()
        if api_batch_size<buffer_batch_size:
            api_batch_size = buffer_batch_size
            if verbose: print('Setting API_BATCH_SIZE to {:,}'.format(api_batch_size))
        batched_df = [ df[i:i+api_batch_size] for i in range(0, len(df), api_batch_size) ]
        idx = 1
        for batch in batched_df:
            print('Processing API request {} of {}'.format(idx, len(batched_df)))
            res5 = self.__predict_api_batch(
                batch, model_id=model_id,
                client_id=client_id,
                idx_var=idx_var,
                categorical_vars=categorical_vars,
                numerical_vars=numerical_vars,
                bool_vars=bool_vars,
                buffer_batch_size=buffer_batch_size,
                verbose=verbose,
                timeout=timeout,
                step=step,
                compressed=compressed,
                staging=staging,
                encoding=encoding,
            )
            res['data2'] = res['data2'].append(res5['data2'])
            idx += 1
        return res

    def __predict_api_batch(self, df, model_id=None, client_id=None, idx_var=None, categorical_vars=[], numerical_vars=[], bool_vars=[], buffer_batch_size=1000, verbose=False, timeout=600, step=2, compressed=False, staging=False, encoding=True):
        """
        :param df:
        :param model_id:
        :param request_id:
        :param client_id:
        :param idx_var:
        :param categorical_vars:
        :param numerical_vars:
        :param buffer_batch_size:
        :param verbose:
        :param compressed:
        :param staging:
        :param encoding:
        :return res5:
        """
        request_id = self._get_request_id()

        if encoding:
            keys = self._keys_load(model_id=model_id, verbose=verbose) # Load encoding keys
            if keys is None:
                print('ERROR! Keys not found. ')
                return None
            data, xref, zref, rref, fref, fref_exp, bref = self._encode(df, keys=keys, categorical_vars=categorical_vars, numerical_vars=numerical_vars, bool_vars=bool_vars, record_id_var=idx_var, verbose=verbose)
            if verbose: print('Total rows encoded: {:,}'.format(len(data)))
        else:
            data = df

        # Save data to buffer
        res = self._buffer_save(
            data,
            client_id=client_id,
            request_id=request_id,
            verbose=verbose,
            batch_size=buffer_batch_size,
            compressed=compressed,
            staging=staging
        )

        # Predict with propensity model and retrieve results
        if res['batches_saved']==res['total_batches']:
            self.__predict(
                request_id=res['request_id'],
                model_id=model_id,
                client_id=client_id,
                idx_field=fref['forward'][idx_var] if encoding else idx_var,
                categorical_fields=[ fref['forward'][var] for var in categorical_vars ] if encoding else categorical_vars,
                verbose=verbose,
                staging=staging
            )
            self._poll(payload={'request_id': res['request_id'], 'client_id': client_id, 'command': 'task-status'}, timeout=timeout, step=step, verbose=verbose)
            data2 = self.__retrieve_predict_results(
                request_id=request_id,
                client_id=client_id,
                verbose=verbose)
        else:
            print('ERROR! Buffer save failed: {}'.format(res))

        # Clear buffer
        res4 = self._buffer_clear(request_id=res['request_id'], client_id=client_id, verbose=verbose)

        # Decode data
        if encoding:
            data2 = self._decode(
                data2,
                categorical_vars=categorical_vars,
                numerical_vars=numerical_vars,
                bool_vars=bool_vars,
                record_id_var=idx_var,
                xref=xref,
                zref=zref,
                rref=rref,
                fref=fref,
                bref=bref,
                verbose=verbose
            )

        # Compile results
        res5 = {}
        res5['data2'] = data2
        res5['model_id'] = model_id
        return res5


    def __predict(self, request_id=None, model_id=None, client_id=None,
                idx_field=None, categorical_fields=[], verbose=False, staging=False):
        """
        :param request_id:
        :param model_id:
        :param client_id:
        :param idx_field:
        :param categorical_fields:
        :param verbose:
        :param staging:
        :return:
        """
        if verbose: print('Predicting propensity model using data in buffer...')
        res = self._client._post(self.__uri, {
            'command': 'propensity-predict',
            'model_id': model_id,
            'request_id': request_id,
            'client_id': client_id,
            'idx_field': idx_field,
            'categorical_fields': categorical_fields,
            'staging': staging,
        })
        return res

    def __retrieve_predict_results(self, request_id=None, client_id=None, verbose=False):
        """
        :param request_id:
        :param client_id:
        :param verbose:
        :return data2:
        """
        data2 = self._buffer_read(request_id=request_id, client_id=client_id, dataframe_name='data2', verbose=verbose)
        return data2

    def check_status(self, model_id=None, client_id=None, verbose=False, encoding=True):
        """
        :param request_id:
        :param client_id:
        :param verbose:
        :param encoding:
        :return res1:
        """
        res1 = {}
        res1['model_id'] = model_id
        res2 = self._status(request_id=model_id, client_id=client_id, verbose=verbose)
        if res2!={} and 'status' in res2.keys():
            res1['status'] = res2['status']
            if res2['status']=='Complete':
                if encoding:
                    # Load encoding keys
                    keys = self._keys_load(model_id=model_id, verbose=verbose)
                    if keys is None:
                        print('ERROR! Keys not found. ')
                        return None
                else:
                    keys = {}
                features, confusion_matrix, stats, roc = self.__retrieve_train_results(
                        request_id=model_id,
                        client_id=client_id,
                        fref=keys['fref_exp'] if encoding else {},
                        verbose=verbose,
                        encoding=encoding)
                res1['features'] = features
                res1['confusion_matrix'] = confusion_matrix
                res1['stats'] = stats
                res1['roc'] = roc
            if res2['status'] in ['Complete', 'Failed']:
                self._buffer_clear(request_id=model_id, client_id=client_id, verbose=verbose)
        return res1

    def train(self, df, client_id=None,
                idx_var=None, outcome_var=None, categorical_vars=[], numerical_vars=[], bool_vars=[],
                algorithm='random-forest-classifier', train_size=0.5, buffer_batch_size=1000,
                verbose=False, timeout=600, step=2, poll=True, smote=False, compressed=False, staging=True, encoding=True):
        """
        :param df:
        :param client_id:
        :param idx_var:
        :param outcome_var:
        :param categorical_vars:
        :param numerical_vars:
        :param algorithm:
        :param train_size:
        :param buffer_batch_size:
        :param verbose:
        :param timeout:
        :param step:
        :param poll:
        :param smote:
        :param compressed:
        :param staging:
        :param encoding:
        :return res5:
        """

        # Encode data
        request_id = self._get_request_id()
        if verbose: print('Model ID: {}'.format(request_id))
        if encoding:
            data, xref, zref, rref, fref, fref_exp, bref = self._encode(df, categorical_vars=categorical_vars, numerical_vars=numerical_vars, bool_vars=bool_vars, record_id_var=idx_var, verbose=verbose)
            self._keys_save(model_id=request_id, keys={'xref': xref, 'zref': zref, 'rref': rref, 'fref': fref, 'fref_exp': fref_exp, 'bref': bref}, verbose=verbose) # Save encoding keys locally
        else:
            data = df

        # Save encoded data to buffer
        res = self._buffer_save(data, client_id=client_id, request_id=request_id, verbose=verbose, batch_size=buffer_batch_size, compressed=compressed, staging=staging)

        # Train propensity model and retrieve results
        if res['batches_saved']==res['total_batches']:
            self.__train(
                request_id=res['request_id'],
                client_id=client_id,
                idx_field=fref['forward'][idx_var] if encoding else idx_var,
                outcome_var=fref['forward'][outcome_var] if encoding else outcome_var,
                categorical_fields=[ fref['forward'][var] for var in categorical_vars ] if encoding else categorical_vars,
                algorithm=algorithm,
                train_size=train_size,
                smote=smote,
                verbose=verbose,
                staging=staging,
            )
            if poll:
                res2 = self._poll(payload={'request_id': res['request_id'], 'client_id': client_id, 'command': 'task-status'}, timeout=timeout, step=step, verbose=verbose)
                if res2['response']['status'] in ['Complete']:
                    features, confusion_matrix, stats, roc = self.__retrieve_train_results(
                        request_id=request_id,
                        client_id=client_id,
                        fref=fref_exp if encoding else {},
                        verbose=verbose,
                        encoding=encoding)
                else:
                    print('WARNING! Training request came back with status: {}'.format(res2['response']['status']))
        else:
            print('ERROR! Buffer save failed: {}'.format(res))

        # Clear buffer
        if poll: res4 = self._buffer_clear(request_id=res['request_id'], client_id=client_id, verbose=verbose)

        # Compile results
        if verbose and not poll: print('Training job started with polling disabled. You will need to request results for this model ID.')
        res5 = {}
        res5['model_id'] = request_id
        if poll:
            res5['features'] = features
            res5['confusion_matrix'] = confusion_matrix
            res5['stats'] = stats
            res5['roc'] = roc
        return res5

    def __train(self, request_id=None, client_id=None,
                idx_field=None, outcome_var=None, categorical_fields=[],
                algorithm='random-forest-classifier', train_size=0.5, smote=False,
                verbose=False, staging=False):
        """
        :param request_id:
        :param client_id:
        :param idx_field:
        :param outcome_var:
        :param categorical_fields:
        :param algorithm:
        :param train_size:
        :param smote:
        :param verbose:
        :param staging:
        :return:
        """
        if verbose: print('Training propensity model using data in buffer...')
        res = self._client._post(self.__uri, {
            'command': 'propensity-train',
            'request_id': request_id,
            'client_id': client_id,
            'algorithm': algorithm,
            'train_size': train_size,
            'smote': smote,
            'idx_field': idx_field,
            'outcome_var': outcome_var,
            'categorical_fields': categorical_fields,
            'staging': staging,
        })
        if verbose: print('Training request posted.')
        return res

    def __retrieve_train_results(self, request_id=None, client_id=None, fref={}, verbose=False, encoding=True):
        """
        :param request_id:
        :param client_id:
        :param fref:
        :param verbose:
        :param encoding:
        :return features:
        :return confusion_matrix:
        :return stats:
        """
        if verbose: print('Retrieving training results...')

        # Features
        if verbose: print('    Retrieving features...')
        features = self._buffer_read(request_id=request_id, client_id=client_id, dataframe_name='features', verbose=verbose)
        # if verbose: print('[__retrieve_train_results] features: {}'.format(features))
        if not features.empty:
            features['Importance'] = features['Importance'].astype('float')
            features.sort_values(by=['Importance'], ascending=False, inplace=True)
            if encoding:
                for idx, row in features.iterrows():
                    features.loc[idx, 'Feature'] = fref_decode_value(features.loc[idx, 'Feature'], fref)
        else:
            if verbose: print('WARNING! Features dataframe is empty')

        # Confusion matrix
        if verbose: print('    Retrieving confusion matrix...')
        confusion_matrix = self._buffer_read(request_id=request_id, client_id=client_id, dataframe_name='confusion_matrix', verbose=verbose)

        # Stats
        if verbose: print('    Retrieving performance stats...')
        stats = self._buffer_read(request_id=request_id, client_id=client_id, dataframe_name='stats', verbose=verbose)
        if not stats.empty:
            stats['Value'] = stats['Value'].astype('float')
        else:
            if verbose: print('WARNING! Stats dataframe is empty')

        # ROC
        if verbose: print('    Retrieving ROC curve...')
        roc = self._buffer_read(request_id=request_id, client_id=client_id, dataframe_name='roc', verbose=verbose)
        if 'TPR' in roc.keys():
            roc['TPR'] = roc['TPR'].astype(float)
        if 'FPR' in roc.keys():
            roc['FPR'] = roc['FPR'].astype(float)
            roc = roc.sort_values(by=['FPR'], ascending=True)

        return features, confusion_matrix, stats, roc
