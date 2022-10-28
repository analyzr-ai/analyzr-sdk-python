import os, sys, time, json
import pandas as pd
from copy import deepcopy

from .runner_base import BaseRunner
from .constants import *
from .utils import *

class PropensityRunner(BaseRunner):
    """
    Run the propensity scoring pipeline

    :param client: SAML SSO client object
    :param base_url: Base URL for the Analyzr API tenant
    """
    def __init__(self, client=None, base_url=None):
        """
        """
        super().__init__(client=client, base_url=base_url)
        self.__uri = '{}/analytics/'.format(self._base_url)
        return

    def predict(
            self, df, model_id=None, client_id=None,
            idx_var=None, categorical_vars=[], numerical_vars=[], bool_vars=[],
            buffer_batch_size=1000, api_batch_size=2000, verbose=False,
            timeout=600, step=2, compressed=False, staging=True, encoding=True):
        """
        Predict probabilities of outcome (propensities) for user-specified
        dataset using a pre-trained model

        :param df: Dataframe containing dataset to be used for training. The data is homomorphically encrypted by the client prior to being transferred to the API buffer when `encoding` is set to `True`
        :param model_id: UUID for a specific model object. Refers to a model that was previously trained
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param idx_var: Name of index field identifying unique record IDs in `df` for audit purposes
        :param categorical_vars: Array of field names identifying categorical fields in the dataframe `df`
        :param numerical_vars: Array of field names identifying categorical fields in the dataframe `df`
        :param bool_vars: Array of field names identifying boolean fields in the dataframe `df`
        :param buffer_batch_size: Batch size for the purpose of uploading data from the client to the server's buffer
        :param api_batch_size: Batch size for the purpose of processing data in the API
        :param verbose: Set to true for verbose output
        :param timeout: Client will keep polling API for a period of `timeout` seconds
        :param step: Polling interval, in seconds
        :param compressed: Perform additional compression when uploading data to buffer
        :param staging: When set to True the API will use temporay secure cloud storage to buffer the data rather than a relational database (default is `True`)
        :param encoding: Encode and decode data with homomorphic encryption
        :return: JSON object with the following attributes:
                    `model_id` (UUID provided with initial request),
                    `data2`: original dataset with cluster IDs appended
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

    def __predict_api_batch(
            self, df, model_id=None, client_id=None, idx_var=None,
            categorical_vars=[], numerical_vars=[], bool_vars=[],
            buffer_batch_size=1000, verbose=False, timeout=600, step=2,
            compressed=False, staging=False, encoding=True):
        """
        :param df:
        :param model_id:
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param idx_var:
        :param categorical_vars:
        :param numerical_vars:
        :param buffer_batch_size: Batch size for the purpose of uploading data from the client to the server's buffer
        :param verbose: Set to true for verbose output
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
            data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                df, keys=keys, categorical_vars=categorical_vars,
                numerical_vars=numerical_vars, bool_vars=bool_vars,
                record_id_var=idx_var, verbose=verbose)
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
            self._poll(
                payload={
                    'request_id': res['request_id'],
                    'client_id': client_id,
                    'command': 'task-status'
                    },
                timeout=timeout,
                step=step,
                verbose=verbose)
            data2 = self.__retrieve_predict_results(
                request_id=request_id,
                client_id=client_id,
                verbose=verbose)
        else:
            print('ERROR! Buffer save failed: {}'.format(res))

        # Clear buffer
        res4 = self._buffer_clear(
            request_id=res['request_id'], client_id=client_id,
            verbose=verbose)

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


    def __predict(
            self, request_id=None, model_id=None, client_id=None,
            idx_field=None, categorical_fields=[], verbose=False,
            staging=False):
        """
        :param request_id:
        :param model_id:
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param idx_field: Name of index field identifying unique record IDs for audit purposes
        :param categorical_fields:
        :param verbose: Set to true for verbose output
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

    def __retrieve_predict_results(
            self, request_id=None, client_id=None,
            verbose=False):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param verbose: Set to true for verbose output
        :return data2:
        """
        data2 = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='data2',
            verbose=verbose)
        return data2

    def check_status(
            self, model_id=None, client_id=None, verbose=False,
            encoding=True):
        """
        Check the status of a specific model run, and retrieve results if model
        run is complete

        :param model_id: UUID for a specific model object
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param verbose: Set to true for verbose output
        :param encoding: Decode results with homomorphic encryption
        :return: JSON object with the following attributes, as applicable:
                    `status` (can be Pending, Complete, or Failed),
                    `features` (table of feature importances),
                    `confusion_matrix` (confusion matrix using test dataset),
                    `stats` (error stats including accuracy, precision, recall, F1, AUC, Gini),
                    `roc` (receiver operating characteristic curve)
        """
        res1 = {}
        res1['model_id'] = model_id
        res2 = self._status(
            request_id=model_id, client_id=client_id,
            verbose=verbose)
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
                self._buffer_clear(
                    request_id=model_id, client_id=client_id,
                    verbose=verbose)
        return res1

    def train(self, df, client_id=None,
            idx_var=None, outcome_var=None, categorical_vars=[], numerical_vars=[],
            bool_vars=[], algorithm='random-forest-classifier', train_size=0.5,
            buffer_batch_size=1000, verbose=False, timeout=600, step=2, poll=True,
            smote=False, param_grid=None, scoring=None, n_splits=None,
            compressed=False, staging=True, encoding=True):
        """
        Train propensity model on user-provided dataset

        :param df: Dataframe containing dataset to be used for training. The data is homomorphically encrypted by the client prior to being transferred to the API buffer when `encoding` is set to `True`
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param idx_var: Name of index field identifying unique record IDs in `df` for audit purposes
        :param outcome_var: Name of dependent variable, usually a boolean variable set to `0` or `1`
        :param categorical_vars: Array of field names identifying categorical fields in the dataframe `df`
        :param numerical_vars: Array of field names identifying categorical fields in the dataframe `df`
        :param bool_vars: Array of field names identifying boolean fields in the dataframe `df`
        :param algorithm: Can be any of the following: `random-forest-classifier`, `gradient-boosting-classifier`, `xgboost-classifier`, `ada-boost-classifier`, `extra-trees-classifier`, `logistic-regression-classifier`. Algorithms are sourced from Scikit-Learn unless otherwise indicated
        :param train_size: Share of training dataset assigned to training vs. testing, e.g. if train_size is set to 0.8 80% of the dataset will be assigned to training and 20% will be randomly set aside for testing and validation
        :param buffer_batch_size: Batch size for the purpose of uploading data from the client to the server's buffer
        :param verbose: Set to true for verbose output
        :param timeout: Client will keep polling API for a period of `timeout` seconds
        :param step: Polling interval, in seconds
        :param poll: Keep polling API while the job is being run (default is `True`)
        :param smote: Apply SMOTE pre-processing
        :param param_grid: TBD
        :param scoring: Scoring methodology to evaluate the performance of the cross-validated model. Common methodologies include `roc_auc`, `accuracy`, and `f1`
        :param n_splits: Number of folds (must be at least 2)
        :param compressed: Perform additional compression when uploading data to buffer
        :param staging: When set to True the API will use temporay secure cloud storage to buffer the data rather than a relational database (default is `True`)
        :param encoding: encode and decode data with homomorphic encryption
        :return: JSON object with the following attributes, as applicable:
                    `model_id` (UUID provided with initial request),
                    `features` (table of feature importances),
                    `confusion_matrix` (confusion matrix using test dataset),
                    `stats` (error stats including accuracy, precision, recall, F1, AUC, Gini),
                    `roc` (receiver operating characteristic curve)
        """

        # Encode data
        request_id = self._get_request_id()
        if verbose: print('Model ID: {}'.format(request_id))
        if encoding:
            data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                df, categorical_vars=categorical_vars,
                numerical_vars=numerical_vars, bool_vars=bool_vars,
                record_id_var=idx_var, verbose=verbose)
            self._keys_save(
                model_id=request_id,
                keys={
                    'xref': xref,
                    'zref': zref,
                    'rref': rref,
                    'fref': fref,
                    'fref_exp': fref_exp,
                    'bref': bref
                },
                verbose=verbose) # Save encoding keys locally
        else:
            data = df

        # Save encoded data to buffer
        res = self._buffer_save(
            data, client_id=client_id, request_id=request_id, verbose=verbose,
            batch_size=buffer_batch_size, compressed=compressed, staging=staging)

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
                param_grid=param_grid,
                scoring=scoring,
                n_splits=n_splits,
            )
            if poll:
                res2 = self._poll(
                    payload={
                        'request_id': res['request_id'],
                        'client_id': client_id,
                        'command': 'task-status'
                    },
                    timeout=timeout,
                    step=step,
                    verbose=verbose)
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
        if poll: res4 = self._buffer_clear(
            request_id=res['request_id'], client_id=client_id,
                verbose=verbose)

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
            param_grid=None, scoring=None, n_splits=None,
            verbose=False, staging=False):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param idx_field: Name of index field identifying unique record IDs for audit purposes
        :param outcome_var:
        :param categorical_fields:
        :param algorithm:
        :param train_size:
        :param smote:
        :param verbose: Set to true for verbose output
        :param staging:
        :param param_grid:
        :param scoring:
        :param n_splits:
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
            'param_grid': param_grid,
            'scoring': scoring,
            'n_splits': n_splits,
        })
        if verbose: print('Training request posted.')
        return res

    def __retrieve_train_results(
            self, request_id=None, client_id=None, fref={}, verbose=False,
            encoding=True):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param fref:
        :param verbose: Set to true for verbose output
        :param encoding:
        :return features:
        :return confusion_matrix:
        :return stats:
        """
        if verbose: print('Retrieving training results...')

        # Features
        if verbose: print('    Retrieving features...')
        features = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='features',
            verbose=verbose)
        if not features.empty:
            features['Importance'] = features['Importance'].astype('float')
            features.sort_values(by=['Importance'], ascending=False, inplace=True)
            if encoding:
                for idx, row in features.iterrows():
                    features.loc[idx, 'Feature'] = fref_decode_value(
                        features.loc[idx, 'Feature'],
                        fref)
        else:
            if verbose: print('WARNING! Features dataframe is empty')

        # Confusion matrix
        if verbose: print('    Retrieving confusion matrix...')
        confusion_matrix = self._buffer_read(
            request_id=request_id, client_id=client_id,
            dataframe_name='confusion_matrix', verbose=verbose)

        # Stats
        if verbose: print('    Retrieving performance stats...')
        stats = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='stats',
            verbose=verbose)
        if not stats.empty:
            stats['Value'] = stats['Value'].astype('float')
        else:
            if verbose: print('WARNING! Stats dataframe is empty')

        # ROC
        if verbose: print('    Retrieving ROC curve...')
        roc = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='roc',
            verbose=verbose)
        if 'TPR' in roc.keys():
            roc['TPR'] = roc['TPR'].astype(float)
        if 'FPR' in roc.keys():
            roc['FPR'] = roc['FPR'].astype(float)
            roc = roc.sort_values(by=['FPR'], ascending=True)

        return features, confusion_matrix, stats, roc
