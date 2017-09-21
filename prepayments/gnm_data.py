#Datastore for GNM II Loan-level data files (which are fixed-width)
import pandas as pd
import numpy as np
import tables
import time
import datetime
import os
import re
import logging
from prepayments.file_defs import *

logger = logging.getLogger()
class DataStore(object):
    """Base class for h5 files based off FNM, GNM, etc. data"""
    DIRECT_ENUMS = {} #enums for which the file stores an index value, not the enum string
    COMPRESSION = tables.Filters(complevel=9,complib='blosc')
    ALL_NINES = re.compile("^9+$")
    def __init__(self, path, read_only=True):
        raise NotImplementedException

    def parse_entry(self, v, k, table_class):
        v_kind = table_class.columns[k].kind
        #self.ALL_NINES.match(v),  I've decided it's better to leave the all
        #9s, even though there'll be overflows for pcts.
        if any([v == '', v.isspace()]):
            if v_kind == 'string':
                v = ''
            elif v_kind in ['int','enum','float']:
                v = 0
            elif v_kind == 'time':
                v = time.mktime(time.strptime("1970-01-01","%Y-%m-%d"))
        elif v_kind == 'int':
            v = int(v)
        elif v_kind == 'float':
            v = float(v)
        elif v_kind == 'enum':
            if k in self.DIRECT_ENUMS: #I.e., enums for which the file stores an index value, not the enum string
                v = int(v)
            else:
                v = table_class.columns[k].enum[v]
        elif v_kind == 'time':
            if self.ALL_NINES.match(v):
                v = time.mktime(time.strptime("1970-01-01","%Y-%m-%d"))
            elif len(v) > 6:
                v = time.mktime(time.strptime(v,"%Y%m%d"))
            else:
                v = time.mktime(time.strptime(v,"%Y%m"))
        return v

    def display_entry(self, p_table, field, v):
        v_kind = p_table.columns[field].kind
        if any([v == '', v.isspace()]):
            if v_kind == 'string':
                v = ''
            elif v_kind in ['int','enum','float']:
                v = 0
            elif v_kind == 'time':
                v = self.dt_from_ts(v)
        elif v_kind in ['float','int']:
            v = '{}'.format(v)
        elif v_kind == 'enum':
            v = p_table.get_enum(field)(v)
        elif v_kind == 'time':
            v = self.dt_from_ts(v)
        return v

    def print_row(self, p_table, index=0, fields=None):
        """Display a table row with nice formatting in columns. This is
        inefficient if you have a lot of rows.

        :param p_table: table from which to display row
        :type p_table: pytables.Table
        :param index: table row to display
        :type index: bigint
        :param fields: which fields to display from table
        :type fields: list"""

        all_fields = p_table.colnames
        if fields is None:
            fields = all_fields

        row = p_table.read(index, index+1)

        return {k:self.display_entry(p_table, k, row[k])}

    def get_samples(self, n=1000):
        raise NotImplementedException

    @staticmethod
    def dt_from_ts(ts):
        """Return a datetime.date given a timestamp as defined in the h5
        file. Fix underflow for post-2038 32bit timestamps."""
        if ts < 0:
            ts = 2**32 + ts #end of epoch is at 2**31-1.
        return datetime.date.fromtimestamp(ts)

    def format_df(self, df, p_table, index_cols=None, parse_enums=True):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, columns=df.dtype.names)
        #Parse datetimes and maybe enums.
        bool_enum = tables.misc.enum.Enum(['N','Y'])
        for k,coltype in p_table.coltypes.items():
            if coltype == 'time32':
                df[k] = [self.dt_from_ts(v) for v in df[k]]
            elif parse_enums and coltype == 'enum':
                p_enum = p_table.get_enum(k)
                if p_enum == bool_enum:
                    df[k] = df[k].astype(bool)
                else:
                    df[k] = [p_enum(v) for v in df[k]]
        if index_cols is not None:
            df.set_index(index_cols, inplace=True)
        return df

    def __exit__(self, exc_type, exc_value, traceback):
        self.h5file.close()

class GNM_II_LL(DataStore, GNM_II_LL_Mixin):
    """Handler for the GNM II loan-level data.

    NB. NaN for pcts is -31073 due to overflow."""

    def __init__(self, path=None, read_only=True):
        """GNM II LL h5 file just contains 3 tables, no hierarchy. One with
        pool-level data, and one with loan-level data, and one tracking
        processed files. There's no need to impose structure in the hierarchy,
        I don't think. Mainly, we're going to be sampling data from the Loans table.

        :param path: Path of the file
        :type path: str
        :param read_only: Is the file locked? Default True for safety
        :type read_only: bool"""

        if path is None:
            path="data/GNM_II_loanlevel.h5"
        self.read_only = read_only
        self.path = path
        if os.path.isfile(path):
            self.h5file = tables.open_file(path,"r" if read_only else "a",
                                          filters=self.COMPRESSION)
            self.pools_table = self.h5file.get_node("/pools")
            self.loans_table = self.h5file.get_node("/loans")
            self.files_table = self.h5file.get_node("/files")
        else:
            if self.read_only:
                logging.warn("Called DataStore with read_only = True, but the path {} doesn't exist.".format(self.path))
                self.read_only = False
            h5file = tables.open_file(path, "a", filters=self.COMPRESSION)
            self.pools_table = h5file.create_table("/", "pools", self.Pools, "Mortgage Pool Information")
            self.loans_table = h5file.create_table("/", 'loans', self.Loans, "Loan Data")
            self.files_table = h5file.create_table("/", 'files', self.Files, "Added files")
            self.h5file = h5file

    def add_file(self, path, force=False):
        """Parse a given GNM II Loan-level data file (which is fixed-width)
        and add it to the h5file."""

        logging.info("Attempting to add {} to DataStore.".format(path))
        VERSIONS = {1.0:datetime.date(2013,2,8),
                    1.1:datetime.date(2013,3,7),
                    1.2:datetime.date(2013,4,15),
                    1.3:datetime.date(2013,5,21),
                    1.4:datetime.date(2013,8,1),
                    1.5:datetime.date(2014,1,1),
                    1.6:datetime.date(2015,4,1)}
        path_dir, filename = os.path.split(path)
        file_pattern = re.compile("GNMA_MBS_LL_MON_([0-9]{4})([0-9]{2})_[0-9]{3}.txt$")
        if not file_pattern.match(filename):
            raise ValueError("File name must match {}. You passed {}".format(
                            file_pattern.pattern, filename))

        if all([filename[:filename.rfind("_")] in
                self.files_table.col("filename"), not force]):
            logging.error("".join(["Tried to add {}, but it's already",
                          " there."]).format(filename))
            return

        year, month = [int(x) for x in file_pattern.match(filename).groups()]
        version = max([k for k,v in VERSIONS.items() if v < datetime.date(year,month,1)])

        with open(path, "r") as f:
            for line in f:
                if line[0] == "L": #Loan-level
                    ans = {k:line.__getslice__(*v) for k,v in
                           self.LOAN_DEF.items()}
                    p_row = self.loans_table.row
                    for k,v in ans.iteritems():
                        p_row[k] = self.parse_entry(v, k, self.Loans)
                    p_row.append()
                elif line[0] == "P": #Pool
                    ans = {k:line.__getslice__(*v) for k,v in
                           self.POOL_DEF.items()}
                    p_row = self.pools_table.row
                    for k,v in ans.iteritems():
                        p_row[k] = self.parse_entry(v, k, self.Pools)
                    p_row.append()
                elif line[0] == "H": #File header
                    ans = {k:line.__getslice__(*v) for k,v in
                           self.FILE_DEF.items()}
                    p_row = self.files_table.row
                    for k,v in ans.iteritems():
                        p_row[k] = self.parse_entry(v, k, self.Files)
                    p_row.append()

                self.h5file.flush()
        logging.info("Added {} to DataStore.".format(filename))

    def get_id_for_index(self, index):
        """Returns DSN and as_of_date (still as timestamp) for a given table index"""
        data = self.loans_table[index]
        return data['dsn'], data['as_of_date']

    def get_samples(self, n=1000, keep_indexes=False):
        """Returns a list of (row_number, cusip, as_of_date) tuples.

        :param n: number of samples to return
        :type n: int
        :param keep_indexes: return the loan_table index of each sample
        :type keep_indexes: bool
        :rtype: pandas.DataFrame"""
        sample_indexes = pd.np.random.choice(self.loans_table.shape[0],n)
        samples = self.loans_table.read_coordinates(sample_indexes)

        samples = self.format_df(samples, self.loans_table,
                              index_cols=['dsn','as_of_date'])
        if keep_indexes:
            samples['index'] = sample_indexes

        return samples

    def format_df(self, df, p_table, index_cols=None, parse_enums=True, blanks=None,
                 zeroes=None, dts=None):
        df = super(GNM_II_LL, self).format_df(df, p_table, index_cols,
                                             parse_enums)
        #Replace 0s with NAs where appropriate.
        if blanks is None:
            blanks = ['origination_type', 'loan_purpose', 'refinance_type', 'removal_reason']
        if zeroes is None:
            zeroes = ['credit_score', 'ltv', 'original_loan_term', 'property_type', 'total_debt_expense_ratio', 'upb', 'upb_at_issuance']
        if dts is None:
            dts = ['loan_origination_date', 'maturity_date']

        for blank in blanks:
            df[blank].where(~(df[blank]=='blank'), np.NaN, inplace=True)
        for zero in zeroes:
            df[zero].where(~(df[zero]==0), np.NaN, inplace=True)
        for dt in dts:
            df[dt].where(~(df[dt]==pd.Timestamp("1970-01-01")), np.NaN, inplace=True)

        return df



class GNM_Pool(DataStore, GNM_Pool_Mixin):
    """Handler for the GNM II pool-level data.

    NB. NaN for pcts is -31073 due to overflow."""

    def __init__(self, path=None, read_only=True):
        """
        :param path: Path of the file
        :type path: str
        :param read_only: Is the file locked? Default True for safety
        :type read_only: bool"""

        self.read_only = read_only
        if path is None:
            path = "data/GNM_II_pool.h5"
        self.path = path
        self.record_classes = (self.Pools, self.MultiIssuers, self.MSAs,
                          self.Years, self.PreModifications, self.Removals,
                          self.States, self.Variouses, self.Supplementals,
                          self.MultiIssuerDeliquencies, self.InsurancePremiums,
                          self.TransferActivities)
        self.record_dicts = (self.POOLS_DEF, self.MULTIISSUERS_DEF, self.MSAS_DEF,
                          self.YEARS_DEF, self.PREMODIFICATIONS_DEF, self.REMOVALS_DEF,
                          self.STATES_DEF, self.VARIOUSES_DEF, self.SUPPLEMENTALS_DEF,
                          self.MULTIISSUERDELIQUENCIES_DEF, self.INSURANCEPREMIUMS_DEF,
                          self.TRANSFERACTIVITIES_DEF)
        self.record_table_names = ('pools', 'multiissuers', 'msas', 'years', 'premodifications', 'removals',
                          'states', 'variouses', 'supplementals', 'multiissuerdeliquencies', 'insurancepremiums',
                          'transferactivities')
        self.record_types = list("DIMOPRSVULFX") #check code

        if os.path.isfile(path):
            self.h5file = tables.open_file(path,"r" if read_only else "a",
                                          filters=self.COMPRESSION)
            for i, tablename in enumerate(self.record_table_names):
                setattr(self, "{}_table".format(tablename),
                        self.h5file.get_node("/{}".format(tablename)))
        else:
            if self.read_only:
                logging.warn("Called DataStore with read_only = True, but the path {} doesn't exist.".format(self.path))
                self.read_only = False
            h5file = tables.open_file(path, "a", filters=self.COMPRESSION)
            for i, tablename in enumerate(self.record_table_names):
                setattr(self, "{}_table".format(tablename),
                        h5file.create_table("/", tablename, self.record_classes[i], tablename))
            #Add candidate pools table
            h5file.create_table("/","candidate_pools",self.CandidatePools, 'candidate_pools')
            self.__add_indexes()
            self.h5file = h5file

    def __add_indexes(self):
        """Add indexes to tables if we're creating from scratch"""
        self.pools_table.cols.cusip.create_index(optlevel=9, kind='full')
        self.pools_table.cols.pool_number.create_index(optlevel=9, kind='full')
        self.variouses_table.cols.cusip.create_index(optlevel=9, kind='full')
        self.msas_table.cols.cusip.create_index(optlevel=9, kind='full')
        self.insurancepremiums_table.cols.pool_number.create_index(optlevel=9, kind='full')
        self.states_table.cols.cusip.create_index(optlevel=9, kind='full')

    def _reindex(self):
        """Redo table indexes after adding data. Shouldn't usually be
        necessary, as autoindex is on."""
        self.pools_table.cols.cusip.reindex(optlevel=9, kind='full')
        self.pools_table.cols.pool_number.reindex(optlevel=9, kind='full')
        self.variouses_table.cols.cusip.reindex(optlevel=9, kind='full')
        self.msas_table.cols.cusip.reindex(optlevel=9, kind='full')
        self.insurancepremiums_table.cols.pool_number.reindex(optlevel=9, kind='full')
        self.states_table.cols.cusip.reindex(optlevel=9, kind='full')

    def add_file(self, path, force=False):
        logging.info("Attempting to add {} to DataStore.".format(path))
        VERSIONS = {1.0:datetime.date(2012,1,1),
                    1.1:datetime.date(2012,5,30),
                    1.2:datetime.date(2012,10,4),
                    1.3:datetime.date(2013,7,1),
                    2.0:datetime.date(2014,12,8),
                    2.1:datetime.date(2015,3,2)}
        path_dir, filename = os.path.split(path)
        monthly_file_pattern = re.compile("monthly_([0-9]{4})([0-9]{2}).txt$")
        weekly_file_pattern = re.compile("weekly_([0-9]{4})([0-9]{2})([0-9]{2}).txt$")
        if monthly_file_pattern.match(filename):
            is_monthly = True
            year, month = [int(x) for x in monthly_file_pattern.match(filename).groups()]
        elif weekly_file_pattern.match(filename):
            is_monthly = False
            year, month, day = [int(x) for x in weekly_file_pattern.match(filename).groups()]
            #Make cusip, as_of hash
            records_hash = [hash(i) for i in zip(self.pools_table.cols.cusip, self.pools_table.cols.as_of_date)]
        else:
            raise ValueError("File name must match {} or {}. You passed {}".format(
                            monthly_file_pattern.pattern, weekly_file_pattern.pattern, filename))

        version = max([k for k,v in VERSIONS.items() if v < datetime.date(year,month,1)])
        file_version = '1.3' if version < 2.1 else '2.1' #These are the only 2
        #I've defined in record_dicts. 2.0 says it has all the changes, but they
        #don't seem to be implemented in the files.


        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i % 10 == 0:
                    print i
                record_type = line[18:19]
                record_ind = self.record_types.index(record_type)
                if not self.record_dicts[record_ind].has_key(file_version):
                    version = '2.1' #default
                else:
                    version = file_version
                ans = {k:line.__getslice__(*v) for k,v in
                       self.record_dicts[record_ind][version].items()}
                p_row = getattr(self, "{}_table".format(self.record_table_names[record_ind])).row
                for k,v in ans.iteritems():
                    p_row[k] = self.parse_entry(v, k, self.record_classes[record_ind])
                if is_monthly:
                    p_row.append()
                else: #check if record already exists for weekly data
                    jim = hash((p_row['cusip'], p_row['as_of_date']))
                    if jim in records_hash:
                        logging.error("Not adding duplicate record at line {}.  {}, {}".format(i,
                                                                                               p_row['cusip'],
                                                                                               p_row['as_of_date']))
                    else:
                        records_hash.append(jim)
                        p_row.append()
            self.h5file.flush()
        logging.info("Added {} to DataStore.".format(filename))

    def update_preprocessed_data(self):
        """Pre-calculate some data

         - Average WAC of all pools per origination month
         - upb of prepaid loans / original balance
        """
        self._update_mean_wacs()
        self._update_gnm2_pools()
#        if self.read_only:
#            raise Exception("Tried to update data on read-only file")

    def _update_gnm2_pools(self):
        """Pool_indicator + pool_type provides a set of codes that will tell
        you if a record is GNM I or GNM II. It could also be used to avoid
        ARMs. Really, we just want GNM II Single family mtgs, which are 'C SF'
        and 'M SF'. This will create a table listing index values for
        qualifying mortgages."""

        p_row = self.h5file.root.candidate_pools.row
        for record in self.pools_table:
            pool_type = ['X','C','M'][record['pool_indicator']] + \
                        record['pool_type']
            if pool_type in ["CSF", "MSF", "CBD", "MFS", "MJM"]:
                p_row['cusip'] = record['cusip']
                p_row['pool_type'] = pool_type
                p_row.append()
        self.h5file.flush()

    def _update_mean_wacs(self):
        """ - Average WAC of all pools per origination month"""
        dts = pd.date_range("1982-04-30","2016-12-31", freq="M")
        wacs = {k:[] for k in dts}
        for record in self.pools_table:
            as_of_date = self.dt_from_ts(record['as_of_date'])
            wala = record['wala']
            if wala < 999:
                wac = record['wac']
                if wac < 99999:
                    origination_date = as_of_date - pd.offsets.MonthBegin(wala)
                    wacs[origination_date].append(record['wac'])
        mean_wacs = pd.Series({k:pd.np.mean(v) for k,v in wacs.iteritems()})
        mean_wacs.to_csv("mean_wacs.csv")

    def _calculate_prior3m_cpr(self):
        if self.h5file.__contains__("/cpr_3m"):
            self.h5file.remove_node("/","cpr_3m")
            self.h5file.flush()
        prior_3m_cprs = self.h5file.create_table("/","cpr_3m", self.CPR_3M)
        prior_3m_cprs.cols.cusip.create_index()
        self.h5file.flush()
        p_row = prior_3m_cprs.row
        for cusip in self.h5file.root.candidate_pools.col('cusip'):
            p_data = self.df_for_cusip(cusip)
            if p_data.shape[0] == 0:
                logging.info("No data for {}".format(cusip))
                continue
            cpr = self.cpr_for_cusip(cusip)
            if cpr is None:
                logging.info("No cpr for {}".format(cusip))
                continue
            cpr = cpr.rolling(3).mean().dropna()
            if cpr.shape[0] == 0:
                logging.info("No cpr for {}".format(cusip))
                continue
            p_row['cusip'] = cusip
            p_row['vintage'] = p_data['pool_issue_date'][0].year
            p_row['security_interest_rate'] = p_data['security_interest_rate'][0]
            for dt, v in cpr.iteritems():
                p_row['date'] = time.mktime(dt.timetuple())
                p_row['cpr_3m'] = v
                p_row.append()
            self.h5file.flush()

    def get_samples(self, n=1000):
        """Returns a list of (row_number, cusip, as_of_date) tuples.

        :param n: number of samples to return
        :type n: int
        :rtype: list"""
        samples = [(i, self.pools_table[i]['cusip'],
                    self.dt_from_ts(self.pools_table[i]['as_of_date']))
                  for i in pd.np.random.choice(self.pools_table.shape[0],n)]
        return samples

    def df_for_cusip(self, cusip, p_table=None, trim=True):
        """Return a DataFrame of data in pools_table for a given cusip

        :param cusip: pool number to look up
        :type cusip: str
        :param p_table: table in which to look up pool number (defaults to
        pools table)
        :type p_table: tables.table.Table
        :param trim: trim the data to good values (i.e. >= 2012-09-01)?
        :type trim: bool

        :rtype pandas.DataFrame:
        """
        if p_table is None:
            p_table = self.pools_table
        p_data = p_table.read_where("cusip == '{}'".format(cusip))
        p_data = self.format_df(p_data, p_table, index_cols=['cusip','as_of_date'])

        if trim:
            p_data = p_data.ix[datetime.date(2012,9,1):,:]
        return p_data

    def smm_for_cusip(self, cusip):
        """Calculate SMM (single month mortality rate) from pools table.

        :param cusip: cusip to look up
        :type cusip: str
        :return: historical SMM
        :rtype: pandas.Series
        """

        p_data = self.df_for_cusip(cusip)
        p_data = p_data[~p_data.index.duplicated(keep='last')]

        #Did it get paid off?

        if p_data.ix[-1, 'pool_upb'] == 0:
            if p_data.shape[0] > 1:
                p_data.loc[p_data.index[-1], 'wala'] = p_data.ix[-2, 'wala'] + 1
            else:
                logger.error("No SMM/CPR data for {}. It has only one row, and that has UPB of 0.".format(cusip))
                return None

        if len(p_data) == 0:
            raise Exception("No data for cusip {}".format(cusip))
        else:
            sb = pd.Series({p_data['as_of_date'][i]:self.scheduled_balance(p_data['pool_upb'][i-1],
                                                                        p_data['security_interest_rate'][i],
                                                                        p_data['wala'][i])
                            for i in xrange(1,p_data.shape[0])})
            smm = sb.sub(p_data.ix[1:,'pool_upb']).div(sb)

            smm.index = pd.DatetimeIndex(smm.index)
        return smm

    def cpr_for_cusip(self, cusip):
        """Calculate CPR from pools table. We prefer BBG's CPR data as it has a
        longer history.

        :param cusip: cusip to look up
        :type cusip: str
        """
        smm = self.smm_for_cusip(cusip)

        if smm is None:
            return None
        else:
            cpr = 1 - (1-smm).pow(12)
            return cpr

    def pool_number_for_cusip(self, cusip):
        p_data = self.pools_table.read_where("cusip == '{}'".format(cusip))
        if len(p_data) > 0:
            pool_number = p_data[0]['pool_number']
        else:
            logger.error("No record in pools table for cusip {}.".format(cusip))
            pool_number = None
        return pool_number

    def _record_exists(self, cusip, dt):
        """Check if a given cusip and as_of_date already exist."""
        where_str = "(cusip == '{}') & (as_of_date == {})"
        if not isinstance(dt,int):
            dt = time.mktime(dt.timetuple())
        p_data = self.pools_table.read_where(where_str.format(cusip, dt))
        return p_data.shape[0] > 0

    @staticmethod
    def scheduled_payment(prior_balance, interest_rate, loan_age,
                             n_payments=360):
      rate = interest_rate / 1200000.
      principal_payment = rate / ((1+rate)**(n_payments -
                                                      loan_age)-1)
      return principal_payment * prior_balance

    @staticmethod
    def scheduled_balance(prior_balance, interest_rate, loan_age,
                          n_payments=360):
        rate = interest_rate / 1200000.
        principal_payment = rate / ((1+rate)**(n_payments - loan_age)-1)
        return prior_balance * (1 - principal_payment)

def main(files):
    ds = GNM_Pool(read_only=False)
    for f in files:
        print 'Adding {} to DataStore {}'.format(f,ds.path)
        ds.add_file(f)

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Add some files.")
    parser.add_argument("files", nargs="+",
                        help="Files to add to the DataStore.")
    args = parser.parse_args()
    sys.exit(main(args.files))

