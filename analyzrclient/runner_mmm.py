from .runner_base import BaseRunner
from .constants import *
from .utils import *

class MMMRunner(BaseRunner):
    """
    Run the marketing mix modeling analysis pipeline

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

    def optimize(self, model_id=None, client_id=None, budget=None, 
                 timeout=600, step=2, verbose=False, encoding=True):
        """
        Optimize marketing mix model

        :param model_id: UUID for a specific model object. Refers to a model
            that was previously trained
        :type model_id: str, required
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :type client_id: string, required
        :param budget: total budget amount to be optimized
        :type budget: float, required
        :param timeout: Client will keep polling API for a period of `timeout`
            seconds
        :type timeout: int, optional
        :param step: Polling interval, in seconds
        :type step: int, optional
        :param verbose: Set to true for verbose output
        :type verbose: boolean, optional
        :param encoding: Encode and decode data with homomorphic encryption
        :type encoding: boolean, optional
        :return: JSON object with the following attributes:
                    `model_id` (UUID provided with initial request),
                    `data2`: original dataset with cluster IDs appended
        """

        request_id = self._get_request_id()

        if encoding is True:

            # Load encoding keys
            keys = self._keys_load(model_id=model_id, verbose=verbose)
            if keys is None:
                print('ERROR! Keys not found. ')
                return None
            xref = keys['xref']
            zref = keys['zref']
            rref = keys['rref']
            fref = keys['fref']
            bref = keys['bref']

        # Optimize marketing mix model and retrieve results
        self.__optimize(
            request_id=request_id, 
            model_id=model_id,
            client_id=client_id, 
            budget=budget, 
            verbose=verbose, 
        )
        self._poll(
            payload={
                'request_id': request_id,
                'client_id': client_id,
                'command': 'task-status'
            },
            timeout=timeout,
            step=step,
            verbose=verbose
        )
        data2 = self.__retrieve_optimize_results(
            request_id=request_id, 
            client_id=client_id, 
            verbose=verbose, 
        )

        # Clear buffer
        res4 = self._buffer_clear(
            request_id=request_id, 
            client_id=client_id,
            verbose=verbose, 
        )

        # Decode data
        if encoding is True:
            for i, val in data2['Parameter'].items():
                if val[:6]=='media_':
                    data2.loc[i, 'Parameter'] = 'media_{}'.format(fref_decode_value(val[6:], fref))

        # Compile results
        res5 = {}
        res5['data2'] = data2
        res5['model_id'] = model_id
        return res5

    def __optimize(self, request_id=None, model_id=None, client_id=None, budget=None, 
                   verbose=False):
        """
        :param request_id:
        :param model_id:
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param budget:
        :param verbose: Set to true for verbose output
        :return:
        """
        if verbose: print('Optimizing marketing mix model...')
        res = self._client._post(self.__uri, {
            'command': 'mmm-optimize',
            'model_id': model_id,
            'request_id': request_id,
            'client_id': client_id, 
            'budget': budget, 
        })
        return res

    def __retrieve_optimize_results(
            self, request_id=None, client_id=None, 
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
                    `train_stats` (training set error stats),
                    `test_stats` (testing set error stats),
                    `lag_stats` (lag analysis stats),
                    `contrib_stats` (contribution analysis stats),
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
                train_stats, test_stats, lag_stats, contrib_stats = self.__retrieve_train_results(
                        request_id=model_id,
                        client_id=client_id,
                        fref=keys['fref_exp'] if encoding else {},
                        zref=keys['zref'] if encoding else {}, 
                        verbose=verbose,
                        encoding=encoding)
                res1['train_stats'] = train_stats
                res1['test_stats'] = test_stats
                res1['lag_stats'] = lag_stats
                res1['contrib_stats'] = contrib_stats
            if res2['status'] in ['Complete', 'Failed']:
                self._buffer_clear(
                    request_id=model_id, client_id=client_id,
                    verbose=verbose)
        return res1

    def train(self, df, client_id=None,
            idx_var=None, time_var=None, outcome_var=None, media_vars=[], other_vars=[], 
            algorithm='mmm-carryover',  
            buffer_batch_size=1000, verbose=False, timeout=600, step=2, poll=True,
            compressed=False, staging=True, encoding=True):
        """
        Train marketing mix model on user-provided dataset

        :param df: Dataframe containing dataset to be used for training. The data
            is homomorphically encrypted by the client prior to being transferred
            to the API buffer when `encoding` is set to `True`
        :type df: DataFrame, required
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :type client_id: string, required
        :param idx_var: Name of index field identifying unique record IDs in
            `df` for audit purposes
        :type idx_var: string, optional
        :param time_var: Name of time field identifying time period of row in `df` 
        :type time_var: string, required
        :param outcome_var: Name of dependent variable
        :type outcome_var: string, required
        :param media_vars: Array of field names identifying media channel
            fields in the dataframe `df`
        :type media_vars: string[], required
        :param other_vars: Array of field names identifying other independent variable
            fields in the dataframe `df`
        :type other_vars: string[], optional
        :param buffer_batch_size: Batch size for the purpose of uploading data
            from the client to the server's buffer
        :type buffer_batch_size: int, optional
        :param algorithm: causal inference algorithm to be used
        :type algorithm: string, optional
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
                    `train_stats` (training set error stats),
                    `test_stats` (testing set error stats),
                    `lag_stats` (lag analysis stats),
                    `contrib_stats` (contribution analysis stats),
        """

        # Encode data
        request_id = self._get_request_id()
        if verbose: print('Model ID: {}'.format(request_id))
        if encoding:
            numerical_vars = media_vars 
            if len(other_vars)>0: numerical_vars = [*numerical_vars, *other_vars]
            data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                df, 
                numerical_vars=numerical_vars, 
                skip_vars=numerical_vars, 
                record_id_var=idx_var, 
                verbose=verbose, 
            )
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
                idx_field=fref['forward'][idx_var] if encoding and idx_var is not None else idx_var,
                time_field=fref['forward'][time_var] if encoding and time_var is not None else time_var,
                outcome_var=fref['forward'][outcome_var] if encoding else outcome_var,
                media_fields=[ fref['forward'][var] for var in media_vars ] if encoding else media_vars,
                other_fields=[ fref['forward'][var] for var in other_vars ] if encoding else other_vars,
                algorithm=algorithm, 
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
                    train_stats, test_stats, lag_stats, contrib_stats = self.__retrieve_train_results(
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
            res5['train_stats'] = train_stats
            res5['test_stats'] = test_stats
            res5['lag_stats'] = lag_stats
            res5['contrib_stats'] = contrib_stats
        return res5

    def __train(self, request_id=None, client_id=None,
            idx_field=None, time_field=None, outcome_var=None, media_fields=[], other_fields=[], 
            algorithm='mmm-carryover', verbose=False, staging=False):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param idx_field: Name of index field identifying unique record IDs for
            audit purposes
        :param time_field:
        :param outcome_var:
        :param categorical_fields:
        :param other_fields:
        :param algorithm:
        :param verbose: Set to true for verbose output
        :param staging:
        :return:
        """
        if verbose: print('Training propensity score matching model using data in buffer...')
        res = self._client._post(self.__uri, {
            'command': 'mmm-train',
            'request_id': request_id,
            'client_id': client_id,
            'idx_field': idx_field,
            'time_field': time_field,
            'outcome_var': outcome_var,
            'media_fields': media_fields,
            'other_fields': other_fields,
            'staging': staging,
            'algorithm': algorithm, 
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
        :return train_stats:
        :return test_stats:
        :return lag_stats:
        :return contrib_stats:
        """
        if verbose: print('Retrieving training results...')

        # Training error stats
        if verbose: print('    Retrieving training error stats...')
        train_stats = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='train_stats_train',
            verbose=verbose)
        if not train_stats.empty:
            train_stats['Value'] = train_stats['Value'].astype('float')
        else:
            if verbose: print('WARNING! Training error stats dataframe is empty')

        # Testing error stats
        if verbose: print('    Retrieving testing error stats...')
        test_stats = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='train_stats_test',
            verbose=verbose)
        if not test_stats.empty:
            test_stats['Value'] = test_stats['Value'].astype('float')
        else:
            if verbose: print('WARNING! Testing error stats dataframe is empty')

        # Lag stats
        if verbose: print('    Retrieving lag stats...')
        lag_stats = self._buffer_read(
            request_id=request_id, client_id=client_id,
            dataframe_name='lag_stats', verbose=verbose)
        for col in lag_stats.columns:
            if col!='index':
                lag_stats[col] = lag_stats[col].astype('float')

        # Contribution stats
        if verbose: print('    Retrieving contribution stats...')
        contrib_stats = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='contrib_stats',
            verbose=verbose)
        for col in contrib_stats.columns:
            if col not in ['stat', 'metric']:
                contrib_stats[col] = contrib_stats[col].astype('float')

        return train_stats, test_stats, lag_stats, contrib_stats 
     

