import os, sys, time, json
import pandas as pd
from copy import deepcopy

from .runner_base import BaseRunner
from .constants import *
from .utils import *

LINEAR_ALGORITHMS = [
    'linear-regression', 
    'bayesian-ridge-regression', 
    'lasso-regression', 
    'ridge-regression', 
]


class RegressionRunner(BaseRunner):
    """
    Run the regression pipeline

    :param client: SAML SSO client object
    :type client: SamlSsoAuthClient, required
    :param base_url: Base URL for the Analyzr API tenant
    :type base_url: str, required
    """
    def __init__(self, client=None, base_url=None):
        """
        """
        super().__init__(client=client, base_url=base_url)
        self.__uri = '{}/analytics/'.format(self._base_url)
        self.__algorithm = None
        self.__includes_carryover_saturation = False
        return

    def predict(self, df, model_id=None, client_id=None,
            idx_var=None, categorical_vars=[], numerical_vars=[], bool_vars=[],
            buffer_batch_size=1000, verbose=False, timeout=600, step=2,
            compressed=False, staging=True):
        """
        Predict outcomes for user-specified dataset using a pre-trained model.
        The data is homomorphically encrypted by the client prior to being
        transferred to the API buffer by default

        :param df: dataframe containing dataset to be used for training.
        :type df: DataFrame, required
        :param model_id: UUID for a specific model object. Refers to a model
            that was previously trained
        :type model_id: str, required
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :type client_id: string, required
        :param idx_var: name of index field identifying unique record IDs in
            `df` for audit purposes
        :type idx_var: string, required
        :param categorical_vars: array of field names identifying categorical
            fields in the dataframe `df`
        :type categorical_vars: string[], required
        :param numerical_vars: array of field names identifying categorical
            fields in the dataframe `df`
        :type numerical_vars: string[], required
        :param bool_vars: array of field names identifying boolean fields in the
            dataframe `df`
        :type bool_vars: string[], optional
        :param buffer_batch_size: batch size for the purpose of uploading data
            from the client to the server's buffer
        :type buffer_batch_size: int, optional
        :param verbose: Set to true for verbose output
        :type verbose: boolean, optional
        :param timeout: client will keep polling API for a period of `timeout`
            seconds
        :type timeout: int, optional
        :param step: polling interval, in seconds
        :type step: int, optional
        :param compressed: perform additional compression when uploading data
            to buffer
        :type compressed: boolean, optional
        :param staging: when set to True the API will use temporay secure cloud
            storage to buffer the data rather than a relational database
            (default is `True`)
        :type staging: boolean, optional
        :return: JSON object with the following attributes:
                    `model_id` (UUID provided with initial request),
                    `data2`: original dataset with cluster IDs appended
        """

        # Load encoding keys
        keys = self._keys_load(model_id=model_id, verbose=verbose)
        if keys is None:
            print('ERROR! Keys not found. ')
            return None
        request_id = self._get_request_id()

        # Encode data and save it to buffer
        data, xref, zref, rref, fref, fref_exp, bref = self._encode(
            df, keys=keys, categorical_vars=categorical_vars,
            numerical_vars=numerical_vars, bool_vars=bool_vars,
            record_id_var=idx_var, verbose=verbose)
        res = self._buffer_save(
            data, client_id=client_id, request_id=request_id, verbose=verbose,
            batch_size=buffer_batch_size, compressed=compressed,
            staging=staging)

        # Predict with regression model and retrieve results
        if res['batches_saved']==res['total_batches']:
            self.__predict(
                request_id=res['request_id'], model_id=model_id,
                client_id=client_id, idx_field=fref['forward'][idx_var],
                categorical_fields=[ fref['forward'][var] for var in categorical_vars ],
                verbose=verbose, staging=staging
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
                request_id=request_id, client_id=client_id, rref=rref,
                verbose=verbose)
        else:
            print('ERROR! Buffer save failed: {}'.format(res))

        # Clear buffer
        res4 = self._buffer_clear(
            request_id=res['request_id'], client_id=client_id,
            verbose=verbose)

        # Decode data
        data2 = self._decode(
            data2, categorical_vars=categorical_vars,
            numerical_vars=numerical_vars, bool_vars=bool_vars,
            record_id_var=idx_var, xref=xref, zref=zref, rref=rref, fref=fref,
            bref=bref, verbose=verbose)

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
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param idx_field: name of index field identifying unique record IDs for
            audit purposes
        :param categorical_fields:
        :param verbose: Set to true for verbose output
        :param staging:
        :return:
        """
        if verbose: print('Predicting regression model using data in buffer...')
        res = self._client._post(self.__uri, {
            'command': 'regression-predict',
            'model_id': model_id,
            'request_id': request_id,
            'client_id': client_id,
            'idx_field': idx_field,
            'categorical_fields': categorical_fields,
            'staging': staging,
        })
        return res

    def __retrieve_predict_results(
            self, request_id=None, client_id=None, rref={},
            verbose=False):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param verbose: Set to true for verbose output
        :return data2:
        """
        data2 = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='data2',
            verbose=verbose)
        return data2

    def check_status(self, model_id=None, client_id=None, verbose=False):
        """
        Check the status of a specific model run, and retrieve results if model
        run is complete. Data is homomorphically encoded by default

        :param model_id: UUID for a specific model object
        :type model_id: str, required
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :type client_id: string, required
        :param verbose: Set to true for verbose output
        :type verbose: boolean, optional
        :return: JSON object with the following attributes, as applicable:
                    `status` (can be Pending, Complete, or Failed),
                    `features` (table of feature importances),
                    `stats` (error stats including R2, p, RMSE, MSE, MAE, MAPE),
        """
        res1 = {}
        res1['model_id'] = model_id
        res2 = self._status(
            request_id=model_id, client_id=client_id,
            verbose=verbose)
        if res2!={} and 'status' in res2.keys():
            res1['status'] = res2['status']
            if res2['status']=='Complete':
                # Load encoding keys
                keys = self._keys_load(model_id=model_id, verbose=verbose)
                if keys is None:
                    print('ERROR! Keys not found. ')
                    return None
                features, stats, coefs, carrysats, elasticities = self.__retrieve_train_results(
                    request_id=model_id,
                    client_id=client_id,
                    fref=keys['fref_exp'],
                    verbose=verbose
                )
                res1['features'] = features
                res1['stats'] = stats
                res1['coefs'] = coefs
                res1['laggingsats'] = carrysats
                res1['elasticities'] = elasticities
            if res2['status'] in ['Complete', 'Failed']:
                self._buffer_clear(
                    request_id=model_id, client_id=client_id,
                    verbose=verbose)
        return res1

    def train(self, df, client_id=None,
            idx_var=None, outcome_var=None, categorical_vars=[], numerical_vars=[], 
            bool_vars=[], saturation_vars=[], lagging_vars=[], 
            algorithm=REGRESSION_DEFAULT_ALGO, train_size=0.5,
            buffer_batch_size=1000, param_grid=None, verbose=False, timeout=600,
            poll=True, step=2, compressed=False, staging=True):
        """
        Train regression model on user-provided dataset

        :param df: dataframe containing dataset to be used for training. The
            data is homomorphically encrypted by the client prior to being
            transferred to the API by default
        :type df: DataFrame, required
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :type client_id: string, required
        :param idx_var: name of index field identifying unique record IDs in
            `df` for audit purposes
        :type idx_var: string, required
        :param outcome_var: name of dependent variable, usually a numerical
            variable
        :type outcome_var: string, required
        :param categorical_vars: array of field names identifying categorical
            fields in the dataframe `df`
        :type categorical_vars: string[], required
        :param numerical_vars: array of field names identifying categorical
            fields in the dataframe `df`
        :type numerical_vars: string[], required
        :param bool_vars: array of field names identifying boolean fields in
            the dataframe `df`
        :type bool_vars: string[], optional
        :param saturation_vars: array of field names identifying fields requiring 
            saturation analysis in the dataframe `df`
        :type bool_vars: string[], optional
        :param lagging_vars: array of field names identifying fields requiring 
            lagging analysis in the dataframe `df`
        :type bool_vars: string[], optional
        :param algorithm: can be any of the following:
            `random-forest-regression`, `gradient-boosting-regression`,
            `xgboost-regression`, `linear-regression-classifier`. Algorithms are
            sourced from Scikit-Learn unless otherwise indicated. Additional
            algorithms may be available
        :type algorithm: string, required
        :param train_size: Share of training dataset assigned to training vs.
            testing, e.g. if train_size is set to 0.8 80% of the dataset will
            be assigned to training and 20% will be randomly set aside for
            testing and validation
        :type train_size: float, optional
        :param buffer_batch_size: batch size for the purpose of uploading data
            from the client to the server's buffer
        :type buffer_batch_size: int, optional
        :param param_grid: Parameter grid to be used during the cross-validation
            grid search (hypertuning). The default is algorithm-specific and set
            by the API.
        :type param_grid: JSON object, optional
        :param verbose: Set to true for verbose output
        :type verbose: boolean, optional
        :param timeout: client will keep polling API for a period of `timeout`
            seconds
        :type timeout: int, optional
        :param poll: keep polling API while the job is being run (default is
            `True`)
        :type poll: boolean, optional
        :param step: polling interval, in seconds
        :type step: int, optional
        :param compressed: perform additional compression when uploading data
            to buffer
        :type compressed: boolean, optional
        :param staging: when set to True the API will use temporay secure cloud
            storage to buffer the data rather than a relational database
            (default is `True`)
        :type staging: boolean, optional
        :return: JSON object with the following attributes, as applicable:
                    `model_id` (UUID provided with initial request),
                    `features` (table of feature importances),
                    `stats` (error stats including accuracy, precision, recall,
                    F1, AUC, Gini),
        """

        self.__algorithm = algorithm

        # Encode data
        request_id = self._get_request_id()
        if verbose: print('Model ID: {}'.format(request_id))
        data, xref, zref, rref, fref, fref_exp, bref = self._encode(
            df, categorical_vars=categorical_vars, numerical_vars=numerical_vars,
            bool_vars=bool_vars, record_id_var=idx_var, verbose=verbose)

        # Save encoding keys locally
        self._keys_save(
            model_id=request_id, keys={'xref': xref, 'zref': zref, 'rref': rref,
            'fref': fref, 'fref_exp': fref_exp, 'bref': bref}, verbose=verbose)

        # Save encoded data to buffer
        res = self._buffer_save(
            data, client_id=client_id, request_id=request_id, verbose=verbose,
            batch_size=buffer_batch_size, compressed=compressed, staging=staging)

        # Train regression model and retrieve results
        if res['batches_saved']==res['total_batches']:
            self.__train(
                request_id=res['request_id'],
                client_id=client_id,
                idx_field=fref['forward'][idx_var],
                outcome_var=fref['forward'][outcome_var],
                categorical_fields=[ fref['forward'][var] for var in categorical_vars ], 
                saturation_fields=[ fref['forward'][var] for var in saturation_vars ],
                lagging_fields=[ fref['forward'][var] for var in lagging_vars ],
                algorithm=algorithm,
                train_size=train_size,
                verbose=verbose,
                staging=staging,
                param_grid=param_grid,
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
                    features, stats, coefs, carrysats, elasticities = self.__retrieve_train_results(
                        request_id=request_id, client_id=client_id,
                        fref=fref_exp, verbose=verbose)
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
            res5['stats'] = stats
            res5['coefs'] = coefs
            res5['laggingsats'] = carrysats
            res5['elasticities'] = elasticities
        return res5

    def __train(self, request_id=None, client_id=None,
            idx_field=None, outcome_var=None, categorical_fields=[],
            saturation_fields=[], lagging_fields=[], 
            algorithm=REGRESSION_DEFAULT_ALGO, train_size=0.5,
            param_grid=None, verbose=False, staging=False):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param idx_field: name of index field identifying unique record IDs for
            audit purposes
        :param outcome_var:
        :param categorical_fields:
        :param saturation_fields:
        :param lagging_fields:
        :param algorithm:
        :param param_grid:
        :param verbose: Set to true for verbose output
        :param staging:
        :return:
        """
        if verbose: print('Training regression model using data in buffer...')
        self.__includes_carryover_saturation = ( len(saturation_fields)>0 or len(lagging_fields)>0 )
        res = self._client._post(self.__uri, {
            'command': 'regression-train',
            'request_id': request_id,
            'client_id': client_id,
            'algorithm': algorithm,
            'train_size': train_size,
            'idx_field': idx_field,
            'outcome_var': outcome_var,
            'categorical_fields': categorical_fields,
            'saturation_fields': saturation_fields, 
            'carryover_fields': lagging_fields, 
            'staging': staging,
            'param_grid': param_grid,
        })
        if verbose: print('Training request posted.')
        return res

    def __retrieve_train_results(
            self, request_id=None, client_id=None, fref={},
            verbose=False):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param fref:
        :param verbose: Set to true for verbose output
        :return features:
        :return stats:
        :return coefs:
        :return carrysats:
        """
        if verbose: print('Retrieving training results...')

        # Features
        if verbose: print('    Retrieving features...')
        features = self._buffer_read(
            request_id=request_id, client_id=client_id,
            dataframe_name='features', verbose=verbose)
        features['Importance'] = features['Importance'].astype('float')
        features.sort_values(by=['Importance'], ascending=False, inplace=True)
        for idx, row in features.iterrows():
            features.loc[idx, 'Feature'] = fref_decode_value(
                features.loc[idx, 'Feature'],
                fref)

        # Stats
        if verbose: print('    Retrieving performance stats...')
        stats = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='stats',
            verbose=verbose)
        stats['Value'] = stats['Value'].astype('float')

        # Coefs
        if self.__algorithm in LINEAR_ALGORITHMS: 
            if verbose: print('    Retrieving coefficients...')
            coefs = self._buffer_read(
                request_id=request_id, client_id=client_id, dataframe_name='coefs',
                verbose=verbose)
            coefs['Value'] = coefs['Value'].astype('float')
            for idx, row in coefs.iterrows():
                coefs.loc[idx, 'Parameter'] = fref_decode_value(
                    coefs.loc[idx, 'Parameter'],
                    fref)
        else:
            coefs = None

        # Saturation and lagging parameters
        if self.__algorithm in LINEAR_ALGORITHMS and self.__includes_carryover_saturation is True: 
            if verbose: print('    Retrieving saturation and lagging parameters...')
            carrysats = self._buffer_read(
                request_id=request_id, client_id=client_id, dataframe_name='carrysats',
                verbose=verbose)
            carrysats['Value'] = carrysats['Value'].astype('float')
            for idx, row in carrysats.iterrows():
                carrysats.loc[idx, 'Variable'] = fref_decode_value(
                    carrysats.loc[idx, 'Variable'],
                    fref)
        else:
            carrysats = None

        # Elasticities
        if self.__algorithm in LINEAR_ALGORITHMS: 
            if verbose: print('    Retrieving elasticities...')
            elasticities = self._buffer_read(
                request_id=request_id, client_id=client_id, dataframe_name='elasticities',
                verbose=verbose)
            elasticities['Value'] = coefs['Value'].astype('float')
            for idx, row in elasticities.iterrows():
                elasticities.loc[idx, 'Parameter'] = fref_decode_value(
                    elasticities.loc[idx, 'Parameter'],
                    fref)
        else:
            elasticities = None


        return features, stats, coefs, carrysats, elasticities
