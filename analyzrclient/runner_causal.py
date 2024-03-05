import os, sys, time, json
import pandas as pd
from copy import deepcopy

from .runner_base import BaseRunner
from .constants import *
from .utils import *

class CausalRunner(BaseRunner):
    """
    Run the causal analysis pipeline

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
        return

    def check_status(
            self, model_id=None, client_id=None, verbose=False,
            encoding=True):
        """
        Check the status of a specific model run, and retrieve results if model
        run is complete

        :param model_id: UUID for a specific model object
        :type model_id: str, required
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :type client_id: string, required
        :param verbose: Set to true for verbose output
        :type verbose: boolean, optional
        :param encoding: Decode results with homomorphic encryption
        :type encoding: boolean, optional
        :return: JSON object with the following attributes, as applicable:
                    `status` (can be Pending, Complete, or Failed),
                    `atx` (average treatment effects),
                    `raw` (dataset stats prior to matching),
                    `misc` (miscellaneous error stats including accuracy, precision, recall,
                    F1, AUC, Gini),
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
                atx, raw, misc, bins = self.__retrieve_train_results(
                        request_id=model_id,
                        client_id=client_id,
                        fref=keys['fref_exp'] if encoding else {},
                        zref=keys['zref'] if encoding else {}, 
                        verbose=verbose,
                        encoding=encoding)
                res1['atx'] = atx
                res1['raw'] = raw
                res1['misc'] = misc
                res1['bins'] = bins
            if res2['status'] in ['Complete', 'Failed']:
                self._buffer_clear(
                    request_id=model_id, client_id=client_id,
                    verbose=verbose)
        return res1

    def train(self, df, client_id=None,
            idx_var=None, outcome_var=None, treatment_var=None, categorical_vars=[], numerical_vars=[], bool_vars=[], 
            algorithm='propensity-score-matching-ci', standard_error=False, 
            buffer_batch_size=1000, verbose=False, timeout=600, step=2, poll=True,
            compressed=False, staging=True, encoding=True):
        """
        Train propensity score matching model on user-provided dataset

        :param df: Dataframe containing dataset to be used for training. The data
            is homomorphically encrypted by the client prior to being transferred
            to the API buffer when `encoding` is set to `True`
        :type df: DataFrame, required
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :type client_id: string, required
        :param idx_var: Name of index field identifying unique record IDs in
            `df` for audit purposes
        :type idx_var: string, required
        :param outcome_var: Name of dependent variable, usually a boolean
            variable set to `0` or `1`
        :type outcome_var: string, required
        :param treatment_var: Name of treatment variable, usually a boolean
            variable set to `0` or `1`
        :type treatment_var: string, required
        :param categorical_vars: Array of field names identifying categorical
            fields in the dataframe `df`
        :type categorical_vars: string[], required
        :param numerical_vars: Array of field names identifying categorical
            fields in the dataframe `df`
        :type numerical_vars: string[], required
        :param bool_vars: Array of field names identifying boolean fields in
            the dataframe `df`
        :type bool_vars: string[], optional
        :param buffer_batch_size: Batch size for the purpose of uploading data
            from the client to the server's buffer
        :type buffer_batch_size: int, optional
        :param algorithm: causal inference algorithm to be used
        :type algorithm: string, optional
        :param standard_error: provide standard error stats for treatment 
            estimates (default is `False`)
        :type standard_error: boolean, optional
        :param verbose: Set to true for verbose output
        :type verbose: boolean, optional
        :param timeout: Client will keep polling API for a period of `timeout`
            seconds
        :type timeout: int, optional
        :param step: Polling interval, in seconds
        :type step: int, optional
        :param poll: Keep polling API while the job is being run (default is
            `True`)
        :type poll: boolean, optional
        :param compressed: Perform additional compression when uploading data to
            buffer. Defaults to `False`
        :type compressed: boolean, optional
        :param staging: When set to True the API will use temporay secure cloud
            storage to buffer the data rather than a relational database.
            Defaults to `True`
        :type staging: boolean, optional
        :param encoding: encode and decode data with homomorphic encryption.
            Defaults to `True`
        :type encoding: boolean, optional
        :return: JSON object with the following attributes, as applicable:
                    `model_id` (UUID provided with initial request),
                    `atx` (average treatment effects),
                    `raw` (dataset stats prior to matching),
                    `misc` (miscellaneous error stats including accuracy, precision, recall,
                    F1, AUC, Gini),
                    `bins` (histogram of matched propensity scores)
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
                treatment_var=fref['forward'][treatment_var] if encoding else treatment_var,
                categorical_fields=[ fref['forward'][var] for var in categorical_vars ] if encoding else categorical_vars,
                algorithm=algorithm, 
                standard_error=standard_error, 
                verbose=verbose,
                staging=staging,
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
                    atx, raw, misc, bins = self.__retrieve_train_results(
                        request_id=request_id,
                        client_id=client_id,
                        fref=fref_exp if encoding else {},
                        zref=zref if encoding else {},
                        outcome_var=outcome_var,
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
            res5['atx'] = atx
            res5['raw'] = raw
            res5['misc'] = misc
            res5['bins'] = bins
        return res5

    def __train(self, request_id=None, client_id=None,
            idx_field=None, outcome_var=None, treatment_var=None, categorical_fields=[],
            algorithm='propensity-score-matching-ci', standard_error=False, 
            verbose=False, staging=False):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param idx_field: Name of index field identifying unique record IDs for
            audit purposes
        :param outcome_var:
        :param treatment_var:
        :param categorical_fields:
        :param algorithm:
        :param standard_error:
        :param verbose: Set to true for verbose output
        :param staging:
        :return:
        """
        if verbose: print('Training propensity score matching model using data in buffer...')
        res = self._client._post(self.__uri, {
            'command': 'causal-train',
            'request_id': request_id,
            'client_id': client_id,
            'idx_field': idx_field,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'categorical_fields': categorical_fields,
            'staging': staging,
            'algorithm': algorithm, 
            'standard_error': standard_error, 
        })
        if verbose: print('Training request posted.')
        return res

    def __retrieve_train_results(
            self, request_id=None, client_id=None, fref={}, zref={}, outcome_var=None,
            verbose=False, encoding=True):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param fref:
        :param zref:
        :param outcome_var:
        :param verbose: Set to true for verbose output
        :param encoding:
        :return atx:
        :return raw:
        :return misc:
        """
        if verbose: print('Retrieving training results...')

        # Average treatment effects
        if verbose: print('    Retrieving treatment effects...')
        atx = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='atx',
            verbose=verbose)
        if not atx.empty:
            atx['Value'] = atx['Value'].astype('float')
        else:
            if verbose: print('WARNING! Treatment effects dataframe is empty')
        if encoding: 
            atx = self.__decode_atx_stats(atx, fref=fref, zref=zref, outcome_var=outcome_var, verbose=verbose)

        # Raw stats
        if verbose: print('    Retrieving raw stats...')
        raw = self._buffer_read(
            request_id=request_id, client_id=client_id,
            dataframe_name='raw', verbose=verbose)
        if encoding: 
            raw = self.__decode_raw_stats(raw, fref=fref, zref=zref, outcome_var=outcome_var, verbose=verbose)
        else:
            for col in raw.columns:
                if col!='variable':
                    raw[col] = raw[col].astype('float')

        # Misc stats
        if verbose: print('    Retrieving miscellaneous stats...')
        misc = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='misc',
            verbose=verbose)
        if not misc.empty:
            misc['Value'] = misc['Value'].astype('float')
        else:
            if verbose: print('WARNING! Miscellaneous stats dataframe is empty')

        # Binned stats
        if verbose: print('    Retrieving binned stats...')
        bins = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='bins',
            verbose=verbose)
        if not bins.empty:
            bins['count_treated'] = bins['count_treated'].astype('int')
            bins['count_untreated'] = bins['count_untreated'].astype('int')
        else:
            if verbose: print('WARNING! Binned stats dataframe is empty')

        return atx, raw, misc, bins 

    def __decode_raw_stats(self, raw, fref={}, zref={}, outcome_var=None, verbose=False):
        """
        Decode raw stats. Note that homomorphic encryption is only homomorphic with averages. 
        Other stats are set to None durign decryption.

        :param raw:
        :param fref:
        :param zref:
        :param outcome_var:
        :param verbose:
        :return raw2:
        """
        if verbose: print('Decoding raw stats...')
        columns = ['control_mean', 'treatment_mean', 'control_stdev', 'treatment_stdev', 'difference_raw', 'difference_normalized', 'variable'] # the order matters!
        raw2 = raw.copy()
        raw2['variable'] = raw2['variable'].apply(lambda s: fref['reverse'][s] if s in fref['reverse'].keys() else s)
        for i in range(0, len(raw2)):
            variable = raw2.iloc[i]['variable']
            if variable in zref.keys():
                for col in columns:
                    if col in ['control_mean', 'treatment_mean']: 
                        raw2.iloc[i][col] = zref_decode(float(raw2.iloc[i][col]), zref[variable])
                    elif col=='difference_raw':
                        raw2.iloc[i][col] = float(raw2.iloc[i]['treatment_mean']) - float(raw2.iloc[i]['control_mean'])
                    elif col!='variable':
                        raw2.iloc[i][col] = None
                    else:
                        pass
            else:
                for col in columns:
                    if col!='variable':
                        val = raw2.iloc[i][col]
                        if val is not None: raw2.iloc[i][col] = float(val) 
        return raw2
    
    def __decode_atx_stats(self, atx, fref={}, zref={}, outcome_var=None, verbose=False):
        """
        Decode average treatment stats. Note that homomorphic encryption is only homomorphic with averages. 
        Other stats are set to None during decryption.

        :param raw:
        :param fref:
        :param zref:
        :param outcome_var:
        :param verbose:
        :return raw2:
        """
        if verbose: print('Decoding treatment stats...')
        atx2 = atx.copy()
        for idx, row in atx2.iterrows():
            if outcome_var in zref.keys():
                parameter = atx2.loc[idx, 'Parameter']
                if parameter in ['atc_matched', 'att_matched', 'ate_matched']: 
                    atx2.loc[idx, 'Value'] = zref_decode(float(atx2.loc[idx, 'Value']), zref[outcome_var])
                else:
                    atx2.loc[idx, 'Value'] = None
            else:
                atx2.loc[idx, 'Value'] = float(atx2.loc[idx, 'Value'])        
        return atx2
