import os, sys, time, json
import pandas as pd
from copy import deepcopy

from .runner_base import BaseRunner
from .constants import *
from .utils import *

DEBUG = True

class ClusterRunner(BaseRunner):
    """
    Run the clustering pipeline

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
            self, request_id=None, client_id=None, verbose=False,
            data=None):
        """
        Check the status of a specific model run. Data is homomorphically
        encoded by default

        :param request_id: UUID for a specific model object
        :type request_id: str, required
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :type client_id: string, required
        :param verbose: Set to true for verbose output
        :type verbose: boolean, optional
        :param data: if data is not None, cluster IDs will be appended and stats
            compiled
        :type data: DataFrame, optional
        :return: JSON object with the following attributes:
                    `status` (can be Pending, Complete, or Failed),
                    `request_id` (UUID provided with initial request),
                    `data` (dataframe with clustering results, if applicable)
        """
        res1 = {}
        res1['request_id'] = request_id
        res2 = self._status(
            request_id=request_id, client_id=client_id,
            verbose=verbose)
        if res2!={} and 'status' in res2.keys():
            res1['status'] = res2['status']
            if res2['status']=='Complete':
                # Load encoding keys
                keys = self._keys_load(model_id=request_id, verbose=verbose)
                if keys is None:
                    print('ERROR! Keys not found. ')
                    return None
                df2 = self.__retrieve_train_results(
                    request_id=request_id, client_id=client_id,
                    categorical_vars=list(keys['xref'].keys()),
                    numerical_vars=list(keys['zref'].keys()),
                    idx_var=keys['idx_var'], xref=keys['xref'],
                    zref=keys['zref'], rref=keys['rref'],
                    fref=keys['fref'], verbose=verbose)
                res1['data'] = df2
                if data is not None:
                    if DEBUG: print('Compiling stats...')
                    res3 = self.__post_process_results(
                        data, df2, keys['idx_var'],
                        list(keys['xref'].keys()))
                    res3['status'] = res1['status']
                    res3['request_id'] = request_id
                    res1 = res3
            if res2['status'] in ['Complete', 'Failed']:
                self._buffer_clear(
                    request_id=request_id, client_id=client_id,
                    verbose=verbose)

        return res1

    def __post_process_results(self, df, pc_id, idx_var, categorical_vars, verbose=False):
        """
        :param df:
        :param pc_id:
        :param idx_var:
        :param categorical_vars:
        :return res:
        """
        res = {}
        if verbose: print(df, pc_id, idx_var)
        df3 = merge_cluster_ids(df, pc_id, idx_var)
        res['data'] = df3
        res['stats'] = compute_cluster_stats(df3, categorical_vars)
        res['distances'] = compute_cluster_distances(df3, categorical_vars)
        return res

    def run(
            self, df, client_id=None, request_id=None, idx_var=None, categorical_vars=[],
            numerical_vars=[], algorithm='pca-kmeans', n_components=5,
            buffer_batch_size=1000, cluster_batch_size=None,
            verbose=False, poll=True, compressed=False, staging=True, out_of_core=False):
        """
        Run clustering algorithm on user-provided dataset

        :param df: dataframe containing dataset to be clustered. The data is
            homomorphically encrypted by the client prior to being transferred
            to the API buffer
        :type df: DataFrame, required
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
        :param algorithm: can be any of the following: `pca-kmeans`,
            `incremental-pca-kmeans`, `pca-kmeans-simple`, `kmeans`,
            `minibatch-kmeans`, `gaussian-mixture`, `birch`, `dbscan`, `optics`,
            `mean-shift`, `spectral-clustering`, `hierarchical-agglomerative`.
            Algorithms are sourced from Scikit-Learn unless otherwise indicated.
        :type algorithm: string, required
        :param n_components: number of clustering components
        :type n_components: int, optional
        :param buffer_batch_size: batch size for the purpose of uploading data
            from the client to the server's buffer :param buffer_batch_size:
            batch size for the purpose of uploading data from the client to the
            server's buffer
        :type buffer_batch_size: int, optional
        :param cluster_batch_size: batch size for the purpose of clustering the
            data provided in the dataframe `df`
        :type cluster_batch_size: int, optional
        :param verbose: Set to true for verbose output
        :type verbose: boolean, optional
        :param poll: keep polling API while the job is being run (default is
            `True`)
        :type poll: boolean, optional
        :param compressed: perform additional compression when uploading data
            to buffer
        :type compressed: boolean, optional
        :param staging: when set to True the API will use temporay secure cloud
            storage to buffer the data rather than a relational database
            (default is `True`)
        :type staging: boolean, optional
        :return: JSON object with the following attributes:
                    `model_id` (UUID provided with initial request),
                    `request_id`: same as `model_id` (provided for backward compatibility),
                    `data`: original dataset with cluster IDs appended
                    `distances`: distance matrix showing inter-cluster distances
                    (centroid to centroid)
                    `stats`: count, frequency, and attribute averages by cluster
                    ID
        """
        request_id = request_id if request_id is not None else self._get_request_id()
        return self.__train(
            df, categorical_vars=categorical_vars, numerical_vars=numerical_vars,
            idx_var=idx_var, buffer_batch_size=buffer_batch_size,
            cluster_batch_size=cluster_batch_size, algorithm=algorithm,
            n_components=n_components, request_id=request_id, client_id=client_id,
            verbose=verbose, compressed=compressed, poll=poll, staging=staging,
            out_of_core=out_of_core
        )

    def __train(
            self, df, categorical_vars=[], numerical_vars=[], bool_vars=[],
            idx_var=None, verbose=False, buffer_batch_size=1000,
            algorithm='pca-kmeans', n_components=5, cluster_batch_size=None,
            request_id=None, client_id=None, timeout=600, step=2,
            compressed=False, poll=True, staging=False, out_of_core=False):
        """
        """
        # Encode data and save it to buffer
        df3 = None
        if verbose: print('Request ID: {}'.format(request_id))
        keys = self._keys_load(model_id=request_id, verbose=True)
        if verbose: print('Keys...', keys)
        data, xref, zref, rref, fref, fref_exp, bref = self._encode(
            df, categorical_vars=categorical_vars, numerical_vars=numerical_vars,
            bool_vars=bool_vars, record_id_var=idx_var, verbose=verbose, keys=keys)

        # Save encoding keys locally 
        self._keys_save(
            model_id=request_id,
            keys={
                'xref': xref,
                'zref': zref,
                'rref': rref,
                'fref': fref,
                'fref_exp': fref_exp,
                'bref': bref,
                'idx_var': idx_var
            },
            verbose=verbose)

        # Save encoded data to buffer
        res = self._buffer_save(
            data, client_id=client_id, request_id=request_id, verbose=verbose,
            batch_size=buffer_batch_size, compressed=compressed, staging=staging)

        # Identify clusters and retrieve results
        if res['batches_saved']==res['total_batches']:
            self.__cluster_train(
                request_id=res['request_id'],
                client_id=client_id,
                idx_field=fref['forward'][idx_var],
                categorical_fields=[ fref['forward'][var] for var in categorical_vars ],
                algorithm=algorithm,
                n_components=n_components,
                batch_size=cluster_batch_size,
                verbose=verbose,
                staging=staging,
                out_of_core=out_of_core,
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
                    df2 = self.__retrieve_train_results(
                        request_id=res['request_id'], client_id=client_id,
                        categorical_vars=categorical_vars,
                        numerical_vars=numerical_vars, idx_var=idx_var,
                        xref=xref, zref=zref, rref=rref, fref=fref,
                        verbose=verbose)
                else:
                    print('WARNING! Training request came back with status: {}'.format(res2['response']['status']))
        else:
            print('ERROR! Buffer save failed: {}'.format(res))

        # Clear buffer
        if poll: res3 = self._buffer_clear(
            request_id=res['request_id'], client_id=client_id,
            verbose=verbose)

        #  Compile results
        if verbose and not poll: print('Clustering job started with polling disabled. You will need to request results for this request ID.')
        res5 = self.__post_process_results(
            df, df2, idx_var, categorical_vars, verbose) if poll else {}
        res5['request_id'] = request_id # append request ID for future reference
        res5['model_id'] = request_id # append request ID for future reference
        return res5

    def __cluster_train(
            self, request_id=None, client_id=None, idx_field=None,
            categorical_fields=[], algorithm='pca-kmeans', n_components=5,
            batch_size=None, verbose=False, staging=False, out_of_core=False):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param idx_field: name of index field identifying unique record IDs for
            audit purposes
        :param categorical_fields:
        :param algorithm:
        :param n_components:
        :param batch_size:
        :param verbose: Set to true for verbose output
        :param staging:
        :return res:
        """
        if verbose: print('Clustering data in buffer...')
        if algorithm=='dask-pca-kmeans' and staging is False:
            print('WARNING! Found staging set to False with dask-pca-kmeans request. Setting staging to True.')
            staging = True
        # uri = '{}/analytics/'.format(self._base_url)
        res = self._client._post(self.__uri, {
            'command': 'cluster-lazy' if algorithm=='dask-pca-kmeans' else 'cluster-train',
            'request_id': request_id,
            'client_id': client_id,
            'algorithm': algorithm,
            'n_components': n_components,
            'batch_size': batch_size,
            'idx_field': idx_field,
            'categorical_fields': categorical_fields,
            'staging': staging,
            'out_of_core': out_of_core, 
        })
        return res

    def __retrieve_train_results(
            self, request_id=None, client_id=None, categorical_vars=[],
            numerical_vars=[], idx_var=None, xref=None, zref=None, rref=None,
            fref=None, verbose=False):
        """
        :return df2:
        """
        df2 = None
        res = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='res',
            verbose=verbose, staging=True)
        if not res.empty:
            df2 = self._decode(
                res, categorical_vars=categorical_vars,
                numerical_vars=numerical_vars, record_id_var=idx_var,xref=xref,
                zref=zref, rref=rref, fref=fref, verbose=verbose)

        return df2

    def predict(
            self, df, model_id=None, client_id=None, idx_var=None,
            categorical_vars=[], numerical_vars=[], buffer_batch_size=1000,
            cluster_batch_size=None, timeout=600, verbose=False,
            compressed=False, staging=True):
        """
        Assign cluster IDs to dataset using a pre-trained model

        :param df: dataframe containing dataset to be clustered. The data is
            homomorphically encrypted by the client prior to being transferred
            to the API buffer
        :type df: DataFrame, required
        :param model_id: UUID for a specific model object
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
        :param buffer_batch_size: batch size for the purpose of uploading data
            from the client to the server's buffer :param buffer_batch_size:
            batch size for the purpose of uploading data from the client to the
            server's buffer
        :type buffer_batch_size: int, optional
        :param cluster_batch_size: batch size for the purpose of clustering the
            data provided in the dataframe `df`
        :type cluster_batch_size: int, optional
        :param timeout: client will keep polling API for a period of `timeout`
            seconds
        :type timeout: int, optional
        :param verbose: Set to true for verbose output
        :type verbose: boolean, optional
        :param compressed: perform additional compression when uploading data to
            buffer
        :type compressed: boolean, optional
        :param staging: when set to True the API will use temporay secure cloud
            storage to buffer the data rather than a relational database
            (default is `True`)
        :type staging: boolean, optional
        :return: JSON object with the following attributes:
                    `model_id` (UUID provided with initial request),
                    `data2`: original dataset with cluster IDs appended
        """
        return self.__predict(
            df, categorical_vars=categorical_vars, numerical_vars=numerical_vars,
            idx_var=idx_var, buffer_batch_size=buffer_batch_size,
            cluster_batch_size=cluster_batch_size, model_id=model_id,
            client_id=client_id, timeout=timeout, verbose=verbose,
            compressed=compressed, staging=staging,
        )

    def __predict(
            self, df, categorical_vars=[], numerical_vars=[], bool_vars=[],
            idx_var=None, verbose=False, buffer_batch_size=1000,
            cluster_batch_size=None, model_id=None, client_id=None, timeout=600,
            step=2, compressed=False, staging=False, encoding=True):
        """
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

        # Predict clusters and retrieve results
        if res['batches_saved']==res['total_batches']:
            self.__cluster_predict(
                request_id=res['request_id'],
                model_id=model_id,
                client_id=client_id,
                idx_field=fref['forward'][idx_var] if encoding else idx_var,
                categorical_fields=[ fref['forward'][var] for var in categorical_vars ] if encoding else categorical_vars,
                batch_size=cluster_batch_size,
                verbose=verbose,
                staging=staging,
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
        #  Compile results
        res5 = {}
        res5['data2'] = data2
        res5['model_id'] = model_id
        return res5

    def __cluster_predict(
            self, request_id=None, model_id=None, client_id=None, idx_field=None,
            categorical_fields=[], batch_size=None, verbose=False, staging=False):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param model_id:
        :param idx_field: name of index field identifying unique record IDs for
            audit purposes
        :param categorical_fields:
        :param batch_size:
        :param verbose: Set to true for verbose output
        :param staging:
        :return res:
        """
        if verbose: print('Predicting cluster IDs using data in buffer...')
        res = self._client._post(self.__uri, {
            'command': 'cluster-predict',
            'model_id': model_id,
            'request_id': request_id,
            'client_id': client_id,
            'batch_size': batch_size,
            'idx_field': idx_field,
            'categorical_fields': categorical_fields,
            'staging': staging,
        })
        return res

    def __retrieve_predict_results(
            self, request_id=None, client_id=None,
            verbose=False):
        """
        :return data2:
        """
        data2 = self._buffer_read(
            request_id=request_id, client_id=client_id, dataframe_name='res',
            verbose=verbose, staging=True)
        return data2
