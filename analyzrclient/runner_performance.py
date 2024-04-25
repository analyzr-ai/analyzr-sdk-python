import os, sys, time, json
import pandas as pd
from copy import deepcopy

from .runner_base import BaseRunner
from .constants import *
from .utils import *

class PerformanceRunner(BaseRunner):
    """
    Run the performance analysis pipeline

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
    
    def train(self, df, client_id=None,
            idx_var=None, time_var=None, outcome_var=None, primary_vars=[], dimensional_vars=[], 
            edges=[], hierarchies=[], udf={}, 
            buffer_batch_size=1000, verbose=False, timeout=600, step=2, poll=True,
            compressed=False, staging=True, encoding=True):
        """
        Train performance analysis model on user-provided dataset

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
        :param time_var: Name of time field identifying date/time period in
            `df` for temporal analysis purposes
        :type time_var: string, required
        :param outcome_var: Name of dependent variable
        :type outcome_var: string, required
        :param primary_vars: Array of field names identifying primary drivers, 
            a.k.a. measures, in the dataframe `df`
        :type primary_vars: string[], required
        :param dimensional_vars: Array of field names identifying dimensional  
            attributes in the dataframe `df`
        :type dimensional_vars: string[], required
        :param edges: Array of pairs of primary drivers identifying causal relationships. 
            For instance `(driver_a, driver_b)` indicates `driver_a` is driven by `driver_b`. 
        :type dimensional_vars: string[], required
        :param hierarchies: dictionary documenting how the dimensional attributes relate to 
            each other. See tutorials and sample code for examples. 
        :type hierarchies: dict, required
        :param udf: user-defined functions documenting how the primary drivers relate to 
            each other, when applicable. See tutorials and sample code for examples. 
        :type udf: dict, optional
        :param buffer_batch_size: Batch size for the purpose of uploading data
            from the client to the server's buffer
        :type buffer_batch_size: int, optional
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
                    `tbd` (placeholder...)
        """
    
        # Encode data
        request_id = self._get_request_id()
        if verbose: print('Model ID: {}'.format(request_id))
        if encoding:
            data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                df, categorical_vars=dimensional_vars,
                numerical_vars=primary_vars, bool_vars=[],
                record_id_var=idx_var, verbose=verbose, numerical=False)
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

        # Train performance analysis model and retrieve results
        if res['batches_saved']==res['total_batches']:
            self.__train(
                request_id=res['request_id'],
                client_id=client_id,
                idx_field=fref['forward'][idx_var] if encoding and idx_var is not None else idx_var,
                time_field=fref['forward'][time_var] if encoding and time_var is not None else time_var,
                outcome_var=fref['forward'][outcome_var] if encoding else outcome_var,
                primary_fields=[ fref['forward'][var] for var in primary_vars ] if encoding else primary_vars,
                dimensional_fields=[ fref['forward'][var] for var in dimensional_vars ] if encoding else dimensional_vars,
                edges=self._encode_edges(edges, fref) if encoding else edges, 
                hierarchies=self._encode_hierarchies(hierarchies, fref) if encoding else hierarchies, 
                udf=self._encode_udf(udf, fref) if encoding else udf, 
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
                    analysis = self.__retrieve_results(
                        request_id=request_id,
                        client_id=client_id,
                        fref=fref if encoding else {},
                        xref=xref if encoding else {},
                        verbose=verbose,
                        encoding=encoding)
                else:
                    print('WARNING! Training request came back with status: {}'.format(res2['response']['status']))
                    analysis = {} 
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
            res5['analysis'] = analysis
        return res5
    
    def __train(self, request_id=None, client_id=None,
            idx_field=None, time_field=None, outcome_var=None, primary_fields=[], dimensional_fields=[],
            edges=[], hierarchies=[], udf={}, 
            verbose=False, staging=False):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param idx_field: Name of index field identifying unique record IDs for
            audit purposes
        :param time_field: 
        :param outcome_var:
        :param primary_fields:
        :param dimensional_fields: 
        :param edges:
        :param hierarchies: 
        :param udf: 
        :param verbose: Set to true for verbose output
        :param staging:
        :return:
        """
        if verbose: print('Training performance analysis model using data in buffer...')
        res = self._client._post(self.__uri, {
            'command': 'analyze-train',
            'request_id': request_id,
            'client_id': client_id,
            'idx_field': idx_field,
            'time_field': time_field,
            'outcome_var': outcome_var,
            'primary_fields': primary_fields,
            'dimensional_fields': dimensional_fields, 
            'edges': edges, 
            'hierarchies': hierarchies, 
            'udf': udf, 
            'staging': staging,
        })
        if verbose: print('Training request posted.')
        return res

    def __retrieve_results(
            self, request_id=None, client_id=None, fref={}, xref={},
            verbose=False, encoding=True):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param fref:
        :param xref:
        :param verbose: Set to true for verbose output
        :param encoding:
        :return analysis:
        """
        if verbose: print('Retrieving training results...')
        analysis = self._buffer_read(
            request_id=request_id, client_id=client_id,
            dataframe_name='res', verbose=verbose, staging=True, dataframe=False)
        if encoding: 
            analysis = {
                'drivers': self.__decode_analysis_driver(analysis['drivers'], fref), 
                'dimensions': self.__decode_analysis_dimension(analysis['dimensions'], fref, xref), 
                'synopsis': {}, 
            }
        return analysis  

    def __decode_analysis_driver(self, obj, fref):
        """
        :param driver:
        :param fref:
        :return obj2:
        """
        if obj=={}: return {}
        obj2 = {
            'node_id': obj['node_id'], 
            'period': obj['period'], 
            'measure': fref['reverse'][obj['measure']], 
            'stats': obj['stats'], 
            'anomaly_detection': obj['anomaly_detection'], 
            'children': [], 
            'main_driver': obj['main_driver'], 
        }
        if obj2['main_driver']['measure'] is not None:
            obj2['main_driver']['measure'] = fref['reverse'][obj2['main_driver']['measure']]
        for child in obj['children']:
            child2 = self.__decode_analysis_driver(child, fref)
            obj2['children'].append(child2)
        return obj2 
    
    def __decode_analysis_dimension(self, obj, fref, xref):
        """
        :param obj:
        :param fref:
        :param xref:
        :return obj2:
        """
        if obj=={}: return {}
        obj2 = {
            'measure': fref['reverse'][obj['measure']], 
            'address': obj['address'], 
            'period': obj['period'],  
            'total': obj['total'], 
            'main_dimension': obj['main_dimension']
        }
        obj2['main_dimension']['dimension'] = fref['reverse'][obj['main_dimension']['dimension']]
        obj2['main_dimension']['member'] = xref[obj2['main_dimension']['dimension']]['reverse'][obj['main_dimension']['member']]
        for dim in obj.keys():
            if dim not in ['measure', 'address', 'period', 'total', 'main_dimension']:
                dim2 = fref['reverse'][dim]
                obj2[dim2] = {}
                for member in obj[dim].keys():
                    member2 = xref[dim2]['reverse'][member] if member!='residual' else member 
                    obj2[dim2][member2] = obj[dim][member]
        return obj2 

    def run(self, df=None, model_id=None, client_id=None,
            idx_var=None, time_var=None, outcome_var=None, address=None, 
            primary_vars=[], dimensional_vars=[], 
            buffer_batch_size=1000, verbose=False, timeout=600, step=2, 
            compressed=False, staging=True, encoding=True):
        """
        Run performance analysis model on user-provided dataset

        :param df: Dataframe containing dataset to be used for training. The data
            is homomorphically encrypted by the client prior to being transferred
            to the API buffer when `encoding` is set to `True`
        :type df: DataFrame, required
        :param model_id: UUID for a specific model object. Refers to a model
            that was previously trained
        :type model_id: str, required
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :type client_id: string, required
        :param idx_var: Name of index field identifying unique record IDs in
            `df` for audit purposes
        :type idx_var: string, optional
        :param time_var: Name of time field identifying date/time period in
            `df` for temporal analysis purposes
        :type time_var: string, required
        :param outcome_var: Name of dependent variable
        :type outcome_var: string, required
        :param address: Name of dimensional address
        :type outcome_var: string, required
        :param primary_vars: Array of field names identifying primary drivers, 
            a.k.a. measures, in the dataframe `df`
        :type primary_vars: string[], required
        :param dimensional_vars: Array of field names identifying dimensional  
            attributes in the dataframe `df`
        :type dimensional_vars: string[], required
        :param buffer_batch_size: Batch size for the purpose of uploading data
            from the client to the server's buffer
        :type buffer_batch_size: int, optional
        :param verbose: Set to true for verbose output
        :type verbose: boolean, optional
        :param timeout: Client will keep polling API for a period of `timeout`
            seconds
        :type timeout: int, optional
        :param step: Polling interval, in seconds
        :type step: int, optional
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
                    `tbd` (placeholder...)
        """
    
        request_id = self._get_request_id()
        refresh_data = ( df is not None and df.empty==False )

        if refresh_data:
            if encoding:
                keys = self._keys_load(model_id=model_id, verbose=verbose)
                if keys is None:
                    print('ERROR! Keys not found. ')
                    return None
                data, xref, zref, rref, fref, fref_exp, bref = self._encode(
                    df, keys=keys, categorical_vars=dimensional_vars,
                    numerical_vars=primary_vars, bool_vars=[],
                    record_id_var=idx_var, verbose=verbose, numerical=False)
            else:
                data = df
            # Save data to buffer
            res = self._buffer_save(
                data, client_id=client_id, request_id=request_id, verbose=verbose,
                batch_size=buffer_batch_size, compressed=compressed, staging=staging)
            if res['batches_saved']!=res['total_batches']:
                print('ERROR! Buffer save failed: {}'.format(res))
        else:
            if encoding:
                keys = self._keys_load(model_id=model_id, verbose=verbose)
                if keys is None:
                    print('ERROR! Keys not found. ')
                    return None
                fref = keys['fref']
                xref = keys['xref']
            else:
                pass # no encoding, no refresh

        # Train performance analysis model and retrieve results
        self.__run(
            request_id=request_id,
            model_id=model_id, 
            client_id=client_id,
            idx_field=fref['forward'][idx_var] if encoding and idx_var is not None else idx_var,
            time_field=fref['forward'][time_var] if encoding and time_var is not None else time_var,
            outcome_var=fref['forward'][outcome_var] if encoding else outcome_var,
            address=self.__encode_address(address, dimensional_vars, fref, xref) if encoding else address, 
            primary_fields=[ fref['forward'][var] for var in primary_vars ] if encoding else primary_vars,
            dimensional_fields=[ fref['forward'][var] for var in dimensional_vars ] if encoding else dimensional_vars,
            verbose=verbose,
            staging=staging, 
            refresh_data=refresh_data, 
        )
        res2 = self._poll(
            payload={
                'request_id': request_id,
                'client_id': client_id,
                'command': 'task-status'
            },
            timeout=timeout,
            step=step,
            verbose=verbose)
        if res2['response']['status'] in ['Complete']:
            analysis = self.__retrieve_results(
                request_id=request_id,
                client_id=client_id,
                fref=fref if encoding else {},
                xref=xref if encoding else {},
                verbose=verbose,
                encoding=encoding)
        else:
            print('WARNING! Job request came back with status: {}'.format(res2['response']['status']))
            analysis = {} 

        # Exit
        if refresh_data: 
            self._buffer_clear(request_id=res['request_id'], client_id=client_id, verbose=verbose)
        return {
            'request_id': request_id,  
            'model_id': model_id,
            'analysis': analysis, 
        }
    
    def __encode_address(self, address, dimensions, fref, xref):
        """
        :param address:
        :param dimensions:
        :param fref:
        :param xref:
        :return address2:
        """
        address2 = []
        idx = 0
        for dim in dimensions: 
            member = xref[dim]['forward'][address[idx]] if address[idx] is not None else None 
            idx += 1
            address2.append(member)
        return tuple(address2)
        

    def __run(self, request_id=None, model_id=None, client_id=None,
            idx_field=None, time_field=None, outcome_var=None, address=None, 
            primary_fields=[], dimensional_fields=[],
            verbose=False, staging=False, refresh_data=False):
        """
        :param request_id:
        :param model_id:
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :param idx_field: Name of index field identifying unique record IDs for
            audit purposes
        :param time_field: 
        :param outcome_var:
        :param primary_fields:
        :param dimensional_fields: 
        :param verbose: Set to true for verbose output
        :param staging:
        :param refresh_data:
        :return:
        """
        if verbose: print('Running performance analysis model using data in buffer...')
        res = self._client._post(self.__uri, {
            'command': 'analyze-run',
            'request_id': request_id,
            'model_id': model_id, 
            'client_id': client_id,
            'idx_field': idx_field,
            'time_field': time_field,
            'outcome_var': outcome_var,
            'address': address, 
            'primary_fields': primary_fields,
            'dimensional_fields': dimensional_fields, 
            'staging': staging,
            'refresh_data': refresh_data, 
        })
        return res

    def purge(self, model_id=None, client_id=None, verbose=False):
        """
        Purge data from performance analysis model 

        :param model_id: UUID for a specific model object. Refers to a model
            that was previously trained
        :type model_id: str, required
        :param client_id: Short name for account being used. Used for reporting
            purposes only
        :type client_id: string, required
        :param verbose: Set to true for verbose output
        :type verbose: boolean, optional
        :return: JSON object with the following attributes, as applicable:
                    `model_id` (UUID provided with initial request),
        """
        if verbose: print('Purging data from performance analysis model...')
        res = self._client._post(self.__uri, {
            'command': 'analyze-purge',
            'model_id': model_id, 
            'client_id': client_id,
        })
        if res['status']!=200:
            print('ERROR! Could not purge model: ', res)
            return None
        return res['response']
    
