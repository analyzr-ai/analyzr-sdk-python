import os, sys, time, json
import pandas as pd
from copy import deepcopy
from io import StringIO
import pickle

from .constants import *
from .utils import *

DEBUG = False

class BaseRunner:
    """
    Base class for all task runners

    """
    def __init__(self, client=None, base_url=None):
        """
        :param client: client object from SamlSsoAuthClient class
        :return:
        """
        self._client = client
        self._base_url = base_url
        return

    def _get_request_id(self):
        """
        :param:
        :return uuid:
        """
        return str(uuid.uuid4())

    def _buffer_save(self, df, batch_size=1000, client_id=None, request_id=None, verbose=False, compressed=False, staging=False):
        """
        :param df:
        :param batch_size:
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param request_id:
        :param verbose: Set to true for verbose output
        :param compressed:
        :param staging:
        :return res:
        """
        if verbose: print('Saving data to buffer...')
        if request_id is None:
            print('WARNING! Request ID is None, aborting buffer save. ')
            return {'request_id': None, 'total_batches': None, 'batches_saved': None}
        batched_df = [ df[i:i+batch_size] for i in range(0, len(df), batch_size) ]
        idx = 0
        success = 0
        for batch in batched_df:
            idx += 1
            for attempt in range(0, 5):
                if self._batch_save(batch, idx, len(batched_df), client_id, request_id, verbose=(attempt==4), compressed=compressed, staging=staging):
                    if verbose: sys.stdout.write('        Processed batch {} of {}\r'.format(idx, len(batched_df)))
                    success += 1
                    break
                else:
                    time.sleep(2)
        if verbose: sys.stdout.write('\n')
        return {'request_id': request_id, 'total_batches': len(batched_df), 'batches_saved': success}

    def _batch_save(self, batch, idx, n, client_id, request_id, verbose=False, compressed=False, staging=False):
        """
        If staging is false, converts dataframe to dictionary and uploads to database buffer.
        If staging is true, converts dataframe to CSV chunk and uploads to data lake buffer.

        :param batch:
        :param idx:
        :param n:
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param request_id:
        :param verbose: Set to true for verbose output
        :param compressed:
        :return success:
        """
        success = False
        uri = '{}/buffer/'.format(self._base_url)
        # if DEBUG: print('[_batch_save] batch #{} len: {} staging: {}'.format(idx, len(batch), staging))
        try:
            res = self._client._post(
                uri,
                {
                    'command': 'upload',
                    'data': batch.to_dict() if not staging else batch.to_csv(header=(idx==1), index=False),
                    'request_id': request_id,
                    'client_id': client_id,
                    'staging': staging,
                },
                compressed=compressed,
            )
            if res['status']==200:
                success = True
            else:
                if verbose: print('[__batch_save] WARNING! Returned status {} for batch {} of {}'.format(res['status'], idx, n))
        except:
            if verbose: print('[__batch_save] WARNING! API call failed for batch {} of {}'.format(idx, n))
        return success

    def _buffer_read(self, client_id=None, request_id=None, dataframe_name='df', verbose=False, raw=False, staging=False):
        """
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param request_id:
        :param verbose: Set to true for verbose output
        :param raw:
        :param staging:
        :return df:
        """
        # if DEBUG: print('[_buffer_read] making call to read buffer')
        res2 = self.__read(client_id=client_id, request_id=request_id, dataframe_name=dataframe_name, verbose=verbose, staging=staging)
        # if DEBUG: print('[_buffer_read] Buffer returned response')
        if raw: return res2
        if res2['status']!=200:
            print('ERROR! Buffer read failed: {}'.format(res2))
            return pd.DataFrame(None)
        # if DEBUG: print('[_buffer_read] Buffer read successfully')
        if res2['response']['data'] is None: return pd.DataFrame()
        return pd.DataFrame(res2['response']['data']) if not staging else pd.read_csv(StringIO(res2['response']['data']))

    def __read(self, client_id=None, request_id=None, dataframe_name='df', verbose=False, staging=False):
        """
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param request_id:
        :param verbose: Set to true for verbose output
        :param staging:
        :return res:
        """
        uri = '{}/buffer/'.format(self._base_url)
        return self._client._post(uri, {
            'command': 'read',
            'request_id': request_id,
            'client_id': client_id,
            'dataframe_name': dataframe_name,
            'staging': staging,
        })

    def queue_purge(self, client_id=None, verbose=False):
        """
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param verbose: Set to true for verbose output
        :return res:
        """
        if verbose: print('Purging queue...')
        uri = '{}/analytics/'.format(self._base_url)
        res = self._client._post(uri, {
            'command': 'purge',
            'client_id': client_id,
        })
        if res['status']!=200: print('WARNING! Queue purge failed: {}'.format(res))
        return res

    def buffer_purge(self, client_id=None, verbose=False):
        """
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param verbose: Set to true for verbose output
        :return res:
        """
        if verbose: print('Purging buffer...')
        uri = '{}/buffer/'.format(self._base_url)
        res = self._client._post(uri, {
            'command': 'purge',
            'client_id': client_id,
        })
        if res['status']!=200: print('WARNING! Buffer purge failed: {}'.format(res))
        return res

    def buffer_usage(self, client_id=None, verbose=False):
        """
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param verbose: Set to true for verbose output
        :return res:
        """
        if verbose: print('Checking buffer usage...')
        uri = '{}/buffer/'.format(self._base_url)
        res = self._client._post(uri, {
            'command': 'usage',
            'client_id': client_id,
        })
        if res['status']!=200: print('WARNING! Checking buffer usage failed: {}'.format(res))
        return res

    def _buffer_clear(self, client_id=None, request_id=None, verbose=False, out_of_core=False):
        """
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param request_id:
        :param verbose: Set to true for verbose output
        :param staging:
        :return res:
        """
        if verbose: print('Clearing buffer with out_of_core...', out_of_core)
        uri = '{}/buffer/'.format(self._base_url)
        res = self._client._post(uri, {
            'command': 'clear',
            'request_id': request_id,
            'client_id': client_id,
            'out_of_core': out_of_core, 
        })
        if res['status']!=200: print('WARNING! Buffer clear failed: {}'.format(res))
        return res

    def _encode(self, df, keys=None, categorical_vars=[], numerical_vars=[], bool_vars=[], record_id_var=None, encode_field_names=True, verbose=False):
        """
        Encode dataframe for the categorical and numerical variables provided.

        :param df:
        :param keys:
        :param categorical_vars:
        :param numerical_vars:
        :param record_id_var:
        :param encode_field_names:
        :param verbose: Set to true for verbose output
        :return df2:
        :return xref: cross-reference dictionary for categorical variables
        :return zref: cross-reference dictionary for numerical variables
        :return rref: cross-reference dictionary for the record ID variable
        :return fref: cross-reference dictionary for field names
        """
        return self._encode_no_keys(
            df,
            categorical_vars=categorical_vars, numerical_vars=numerical_vars, bool_vars=bool_vars, record_id_var=record_id_var,
            encode_field_names=encode_field_names, verbose=verbose,
        ) if keys is None else self._encode_with_keys(
            df, keys,
            categorical_vars=categorical_vars, numerical_vars=numerical_vars, bool_vars=bool_vars, record_id_var=record_id_var,
            encode_field_names=encode_field_names, verbose=verbose,
        )

    def _encode_no_keys(self, df, categorical_vars=[], numerical_vars=[], bool_vars=[], record_id_var=None, encode_field_names=True, verbose=False):
        """
        Encode dataframe for the categorical and numerical variables provided. This
        assumes no pre-existing encoding, returns encoded data and matching
        encoding keys.

        :param df:
        :param categorical_vars:
        :param numerical_vars:
        :param record_id_var:
        :param encode_field_names:
        :param verbose: Set to true for verbose output
        :return df2:
        :return xref: cross-reference dictionary for categorical variables
        :return zref: cross-reference dictionary for numerical variables
        :return rref: cross-reference dictionary for the record ID variable
        :return fref: cross-reference dictionary for field names
        """
        df2 = deepcopy(df)

        # Encode categorical variables
        if verbose: print('Encoding categorical variables:')
        xref = {}
        for col in categorical_vars:
            if verbose: print('\t{}'.format(col))
            df2[col], xref[col] = xref_encode(df[col])

        # Encode numerical variables
        if verbose: print('Encoding numerical variables:')
        zref = {}
        for col in numerical_vars:
            if verbose: print('\t{}'.format(col))
            df2[col], zref[col] = zref_encode(df[col])

        # Encode boolean variables
        # if verbose: print('Encoding boolean variables:')
        bref = {}
        for col in bool_vars:
            if verbose: print('\t{}'.format(col))
            df2[col], bref[col] = bref_encode(df[col])

        # Encode record ID
        if record_id_var is not None:
            if verbose: print('Encoding record IDs...')
            df2, rref = rref_encode(df2, record_id_var)
        else:
            rref = {}

        # Encode field names
        if encode_field_names:
            if verbose: print('Encoding field names...')
            df2, fref = fref_encode(df2)
            fref_exp = fref_to_fref_expanded(fref, xref)

        return df2, xref, zref, rref, fref, fref_exp, bref

    def _encode_with_keys(self, df, keys, categorical_vars=[], numerical_vars=[], bool_vars=[], record_id_var=None, encode_field_names=True, verbose=False):
        """
        Encode dataframe for the categorical and numerical variables provided. This
        assumes pre-existing encoding keys are provided, returns encoded data and
        encoding keys.

        Note: record IDs are re-encoded every time with new keys to accommodate
        additional records.

        :param df:
        :param keys:
        :param categorical_vars:
        :param numerical_vars:
        :param record_id_var:
        :param encode_field_names:
        :param verbose: Set to true for verbose output
        :return df2:
        :return xref: cross-reference dictionary for categorical variables
        :return zref: cross-reference dictionary for numerical variables
        :return rref: cross-reference dictionary for the record ID variable
        :return fref: cross-reference dictionary for field names
        """
        df2 = deepcopy(df)

        # Encode categorical variables
        if verbose: print('Encoding categorical variables:')
        xref = keys['xref']
        for col in categorical_vars:
            if verbose: print('\t{}'.format(col))
            df2[col] = xref_encode_with_keys(df[col], xref[col])

        # Encode numerical variables
        if verbose: print('Encoding numerical variables:')
        zref = keys['zref']
        for col in numerical_vars:
            if verbose: print('\t{}'.format(col))
            df2[col] = zref_encode_with_keys(df[col], zref[col])

        # Encode boolean variables
        # if verbose: print('Encoding boolean variables:')
        bref = keys['bref']
        for col in bool_vars:
            if verbose: print('\t{}'.format(col))
            df2[col] = bref_encode_with_keys(df[col], bref[col])

        # Encode record ID
        if record_id_var is not None:
            if verbose: print('Encoding record IDs...')
            # rref = keys['rref']
            # df2 = rref_encode_with_keys(df2, record_id_var, rref)
            df2, rref = rref_encode(df2, record_id_var)
        else:
            rref = {}

        # Encode field names
        if encode_field_names:
            if verbose: print('Encoding field names...')
            fref = keys['fref']
            fref_exp = keys['fref_exp']
            df2 = fref_encode_with_keys(df2, fref)
        else:
            fref = {}
            fref_exp = {}

        df2 = df2.dropna()

        return df2, xref, zref, rref, fref, fref_exp, bref

    def _decode(self, df, categorical_vars=[], numerical_vars=[], bool_vars=[], record_id_var=None, xref={}, zref={}, rref={}, fref={}, bref={}, verbose=False):
        """
        Decode dataframe using the encoding keys provided (xref, zrfef, rref, fref)

        :param df:
        :param categorical_vars:
        :param numerical_vars:
        :param record_id_var:
        :param xref: cross-reference dictionary for categorical variables
        :param zref: cross-reference dictionary for numerical variables
        :param rref: cross-reference dictionary for the record ID variable
        :param fref: cross-reference dictionary for field names
        :param verbose: Set to true for verbose output
        :return df2:
        """
        if df is None or df.empty: return df
        df2 = deepcopy(df)

        # Decode field names
        if verbose: print('Decoding field names...')
        df2 = fref_decode(df2, fref) # decode field names first

        # Decode categorical variables
        if verbose: print('Decoding categorical variables:')
        for col in categorical_vars:
            if col in df2.columns:
                if verbose: print('\t{}'.format(col))
                df2[col] = xref_decode(df2[col], xref[col])

        # Decode numerical variables
        if verbose: print('Decoding numerical variables:')
        for col in numerical_vars:
            if col in df2.columns:
                if verbose: print('\t{}'.format(col))
                df2[col] = zref_decode(df2[col].astype('float'), zref[col])

        # Decode boolean variables
        # if verbose: print('Decoding boolean variables:')
        for col in bool_vars:
            if col in df2.columns:
                if verbose: print('\t{}'.format(col))
                df2[col] = bref_decode(df2[col].astype('int'), zref[col])

        # Decode record ID
        if record_id_var is not None and record_id_var in df2.columns:
            if verbose: print('Decoding record IDs...')
            df2 = rref_decode(df2, record_id_var, rref, verbose=True)

        return df2

    def _poll(self, payload={}, timeout=600, step=1, verbose=False):
        """
        :param payload:
        :param timeout:
        :param step:
        :param verbose: Set to true for verbose output
        :return res:
        """
        counter = 0
        res = {}
        uri = '{}/analytics/'.format(self._base_url)
        while counter<timeout:
            time.sleep(step)
            try:
                res = self._client._post(uri, payload)
                if verbose: sys.stdout.write('[_poll][{}] {}\r'.format(counter, res))
                if res['response']['status'] in ['Complete', 'Failed'] or 'Failed:' in res['response']['status']: break
            except:
                pass
            counter += step
        if verbose: sys.stdout.write('\n')
        return res

    def _status(self, request_id=None, client_id=None, verbose=False):
        """
        :param request_id:
        :param client_id: Short name for account being used. Used for reporting purposes only
        :param verbose: Set to true for verbose output
        :return res:
        """
        res = {}
        uri = '{}/analytics/'.format(self._base_url)
        if request_id is None or client_id is None:
            print('[_status] ERROR! Invalid status request (request_id: {}, client_id: {})'.format(request_id, client_id))
        else:
            try:
                res = self._client._post(uri, {'request_id': request_id, 'client_id': client_id, 'command': 'task-status'})
            except:
                if verbose: print('[_status] ERROR! Could not retrieve status for request ID: {} and client ID: {}'.format(request_id, client_id))
            else:
                if res['status']==200:
                    res = res['response']
                else:
                    if verbose: print('[_status] ERROR! Status inquiry for request ID: {} and client ID: {} returned status: {}'.format(request_id, client_id, res['status']))
                    res = {}
        return res

    def _keys_save(self, model_id=None, keys=None, verbose=False):
        """
        :param model_id:
        :param keys:
        :param verbose: Set to true for verbose output
        :return:
        """
        if verbose: print('Saving encoding keys locally...')
        if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
        filename = '{}/{}.bin'.format(TEMP_DIR, model_id)
        file = open(filename, 'wb')
        pickle.dump(keys, file)
        file.close()
        return

    def _keys_load(self, model_id=None, verbose=False):
        """
        :param model_id:
        :param verbose: Set to true for verbose output
        :return keys:
        """
        if verbose: print('Loading encoding keys...')
        try:
            filename = '{}/{}.bin'.format(TEMP_DIR, model_id)
            file = open(filename, 'rb')
            keys = pickle.load(file)
            file.close()
        except FileNotFoundError:
            print('ERROR! Keys not found for model_id: {}'.format(model_id))
            keys = None

        return keys
