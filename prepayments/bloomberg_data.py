#Get bloomberg data. Particularly house prices.
from bloomberg import Bloomberg
import pandas as pd
import datetime
import numpy as np
from numpy import NaN
import logging
logger = logging.getLogger()

class BBG_Data(object):
    """BBG_Data contains a pandas HDFStore, since it'll only be storing
    DataFrames and Panels."""
    DEFAULT_START_DATE = datetime.date(1980,1,1)

    def __init__(self,path="data/bbg_data.h5", read_only=True):
        self.read_only = read_only
        self.path = path
        self.bbg = Bloomberg()
        self.hdfstore = pd.HDFStore(path, mode='r' if read_only else 'a',
                                    compression=0, complib='blosc')

    def get_all_data(self):
        if self.read_only:
            error_str = "Attempted to get all data in {} for {}, but file is opened read-only"
            error_str = error_str.format(self.path, type(self))
            logging.error(error_str)
            raise Exception(error_str)

    def update_data(self):
        if self.read_only:
            error_str = "Attempted to update data in {} for {}, but file is opened read-only"
            error_str = error_str.format(self.path, type(self))
            logging.error(error_str)
            raise Exception(error_str)

class HPA_Data(BBG_Data):
    """Datastore for Home Price Appreciation

        Where possible, we have MSA-level data. Elsewhere, we have state-level,
        but that only starts in 2010. Before 2010, I'll have to use the 20-City
        Composite from the S&P MSA-level data."""
    STATE_START_DATE = datetime.date(2010,1,1)
    MSA_TICKS ={'SPCS{} Index'.format(k):v for k,v in {
        '20SA':'20-City Composite', 'USS':'U.S. National', 'NYS':'New York',
        'SFS':'San Francisco', 'WDCS':'Washington D.C.', 'SEAS':'Seattle',
        'BOSS':'Boston', 'CHAS':'Charlotte', 'LVS':'Las Vegas',
        'MIAS':'Miami', 'LAS':'Los Angeles', 'DALS':'Dallas',
        'CHIS':'Chicago', 'MINS':'Minneapolis', 'SDS':'San Diego',
        'CLES':'Cleveland', 'PHXS':'Phoenix', 'ATLS':'Atlanta',
        'DENS':'Denver', 'DETS':'Detroit', 'TMPS':'Tampa',
        'PORS':'Portland'}.items()}
    STATE_TICKS = {'MEHP{} Index'.format(k):v for k,v in {
        'WAON':'WA','NJEY':'NJ','NYRK':'NY','AKKA':'AK','AKSS':'AR','ALMA':'AL',
        'AZNA':'AZ','CLFN':'CA','CODO':'CO','CTUT':'CT','DCIA':'DC','DWLR':'DE',
        'FLDA':'FL','GAIA':'GA','HIII':'HI','IAWA':'IA','IDAO':'ID','ILIS':'IL',
        'INNA':'IN','KSAS':'KS','KYKY':'KY','LANA':'LA','MATS':'MA','MDND':'MD',
        'MENE':'ME','MIAN':'MI','MSOT':'MN','MSPI':'MS','MSUR':'MO','MTNA':'MT',
        'NCNA':'NC','NDTA':'ND','NEKA':'NE','NMXO':'NM','NVDA':'NV','NWHS':'NH',
        'OHIO':'OH','OKMA':'OK','ORON':'OR','PAIA':'PA','RDIL':'RI','SCNA':'SC',
        'SDTA':'SD','TNEE':'TN','TXAS':'TX','UTAH':'UT','VAIA':'VA','VTNT':'VT',
        'WIIN':'WI','WVIA':'WV','WYNG':'WY'}.items()}
    MSA_CONCORDANCE = pd.read_hdf("data/msa_concordance.h5", "table")

    def __init__(self, read_only=True):
        super(HPA_Data, self).__init__(read_only=read_only)
        self.state_data = self.hdfstore['state'] #cache it
        self.cpi = self.hdfstore['cpi']
        self.msa = self.hdfstore['msa']

    @staticmethod
    def _dt_from_float(flt):
        """Convert a float date (like 1995.12500) to a Timestamp"""
        ts = datetime.date(int(flt),1,1) + pd.tseries.offsets.Day(365 * (flt % 1))
        return ts + pd.tseries.offsets.MonthEnd()

    def get_all_data(self, start_date=datetime.date(1952,12,31)):
        """MSA-based is available since 2000, but incomplete. State-based is only
        since 2010. We'll need some combination of the 2, plus the US National
        index. U.S. National is supplemented by older data form Shiller's
        website."""
        super(HPA_Data, self).get_all_data()
        old_data = pd.read_excel("http://www.econ.yale.edu/~shiller/data/Fig3-1.xls",
                      sheetname='Data', header=7).ix[63:506,7:9]
        old_data.index = [self._dt_from_float(x) for x in
                          old_data.pop("Date.2")]
        old_data.columns = self.MSA_TICKS['USS']
        msa_rawdata = self.bbg.get_history(self.MSA_TICKS,
                                           start_date=start_date).rename(columns=self.MSA_TICKS)
        state_rawdata = self.bbg.get_history(self.STATE_TICKS,
                                             start_date=start_date).rename(columns=self.STATE_TICKS)
        cpi_rawdata = self.bbg.get_history("CPI INDX Index", start_date=start_date).ix[:,0]

        self.hdfstore['msa'] = msa_rawdata.asfreq(freq="M", method='pad').combine_first(old_data)
        self.hdfstore['state'] = state_rawdata.asfreq(freq="M", method='pad')
        self.hdfstore['cpi'] = pd.np.log(cpi_rawdata).diff().asfreq(freq="M", method='pad')

    def update_data(self):
        super(HPA_Data, self).update_data()
        msa_rawdata = self.hdfstore['msa']
        state_rawdata = self.hdfstore['state']
        cpi_rawdata = self.hdfstore['cpi']

        new_msa_rawdata = self.bbg.get_history(self.MSA_TICKS,
                                           start_date=msa_rawdata.index[-1]).rename(columns=self.MSA_TICKS)
        new_state_rawdata = self.bbg.get_history(self.STATE_TICKS,
                                             start_date=state_rawdata.index[-1]).rename(columns=self.STATE_TICKS)
        new_cpi_rawdata = self.bbg.get_history("CPI INDX Index",
                                               start_date=cpi_rawdata.index[-1])
        new_cpi_rawdata = pd.np.log(new_cpi_rawdata).asfreq(freq="M",
                                                            method='pad').diff()

        self.hdfstore['msa'] = msa_rawdata.combine_first(new_msa_rawdata.asfreq(freq="M", method='pad'))
        self.hdfstore['state'] = state_rawdata.combine_first(new_state_rawdata.asfreq(freq="M", method='pad'))
        self.hdfstore['cpi'] = cpi_rawdata.combine_first(new_cpi_rawdata)

    def get_returns_for_msa_code(self, msa_code):
        """Return the best return set of real price appreciation for a given msa_code.

        Pass 'None' to just get the US National rate.

        NB: THESE ARE IN LOG SPACE"""
        if isinstance(msa_code, str):
            logger.warning("Passed string for msa_code, when I expected an int.")
            msa_code = int(msa_code)

        if msa_code is None:
            logger.info("Returning generic HPA data")
            data = np.log(self.msa['U.S. National']).diff()
        else:
            msa_info = self.MSA_CONCORDANCE.loc[msa_code]
            tick = self.__df_to_single_value(msa_info, 'Index_Ticker')
            if isinstance(tick, str):
                data = np.log(self.msa[self.MSA_TICKS[tick]]).diff()
            else: # Will have to call states 
                tick = self.__df_to_single_value(msa_info, 'Short_State')
                data = self.get_returns_for_state(tick)
        if data is not None:
            data = data.sub(self.cpi)
        return data

    def get_returns_for_state(self, state):
        if state not in self.state_data.columns:
            logging.info("No state data")
            data = None
        else:
            data = pd.np.log(self.state_data[state]).diff().sub(self.cpi)
            #Fill in earlier values
            data = data.combine_first(
                self.get_returns_for_msa_code(msa_code=None))
        return data

    @staticmethod
    def __df_to_single_value(df, fld):
        if isinstance(df, pd.DataFrame):
            ans = df[fld].dropna()
            ans = np.NaN if len(ans) == 0 else ans.values[0]
        else:
            ans = df[fld]
        return ans

class YC_Data(BBG_Data):
    """Datastore for Yield Curves

        10Y and 2Y US Generic Bond rates"""
    YC_TICKS = {'USGG10Y Index':'10Y', 'USGG2YR Index':'2Y'}

    def __init__(self, read_only=True):
        super(YC_Data, self).__init__(read_only=read_only)
        self.yc = self.hdfstore['yc'] #Cache it.

    def get_all_data(self, start_date=BBG_Data.DEFAULT_START_DATE):
        super(YC_Data, self).get_all_data()
        yc_rawdata = self.bbg.get_history(self.YC_TICKS,
                                          start_date=start_date).rename(columns=self.YC_TICKS)
        self.hdfstore['yc'] = yc_rawdata.asfreq(freq="M", method='pad')

    def update_data(self):
        super(YC_Data, self).update_data()
        yc_rawdata = self.hdfstore['yc']

        new_yc_rawdata = self.bbg.get_history(self.YC_TICKS,
                                          start_date=yc_rawdata.index[-1]).rename(columns=self.YC_TICKS)
        self.hdfstore['yc'] = yc_rawdata.combine_first(new_yc_rawdata.asfreq(freq="M", method='pad'))

    def get_spread(self, dt, origination_dt):
        """Change in YC between now and origination"""
        yc = self.yc.ix[[origination_dt, dt],['2Y', '10Y']]
        ans = yc.diff(axis=1).diff().ix[1,1]
        return ans

    def fetch_dt(self, dt):
        """Get a specific (usually non-month-end) date"""
        if dt in self.yc.index:
            new_yc_rawdata = self.yc.ix[dt]
        else:
            new_yc_rawdata = self.bbg.get_history(self.YC_TICKS, start_date=dt, end_date=dt,
                                                  other_params=dict(periodicitySelection='DAILY')).rename(columns=self.YC_TICKS)
            self.hdfstore['yc'] = self.yc.combine_first(new_yc_rawdata)
            self.hdfstore.flush()
            self.yc = self.hdfstore['yc']
        return new_yc_rawdata


class Mortgage_Rate_Data(BBG_Data):
    """Prevailing mortgage rates by month.

    15Y starts in '98, so we need to fill in."""
    RATE_TICKS = {'NMCMFUS Index':'30Y', 'NMCM15US Index':'15Y'}

    def __init__(self, read_only=True):
        super(Mortgage_Rate_Data, self).__init__(read_only=read_only)
        self.mtg_rates = self.hdfstore['mtg_rates'] #Cache it.

    def get_all_data(self, start_date=BBG_Data.DEFAULT_START_DATE):
        super(Mortgage_Rate_Data, self).get_all_data()
        mr_rawdata = self.bbg.get_history(self.RATE_TICKS,
                                          start_date=start_date).rename(columns=self.RATE_TICKS)
        #Was going to Backfill 15Y with 15-30 spread on generic gov rates and
        #30Y mtg data, but there's actually not much of a spread anyway (between
        #30Y and 10Y back then, as there's no 15Y generic. I'm just going to
        #fill with 30Y mtg rate.
        mr_rawdata.ix[:'1991','15Y'] = mr_rawdata.ix[:'1991','15Y'].combine_first(
                                        mr_rawdata.ix[:'1991','30Y'])
        self.hdfstore['mtg_rates'] = mr_rawdata.asfreq(freq="M", method='pad')

    def update_data(self):
        super(Mortgage_Rate_Data, self).update_data()
        mr_rawdata = self.hdfstore['mtg_rates']

        new_mr_rawdata = self.bbg.get_history(self.RATE_TICKS,
                                          start_date=mr_rawdata.index[-1]).rename(columns=self.RATE_TICKS)
        self.hdfstore['mtg_rates'] = mr_rawdata.combine_first(new_mr_rawdata.asfreq(freq="M", method='pad'))

    def get_rate_for_loan_age(self, loan_age, dt, pad=False):
        """Get a weighted mortgage rate that's some combination of 30Y and 15Y
        prevailing mortgage rates based on the loan age.

        JPM's function looks like a sigmoid transform."""
        if dt.day == 1:
            logger.info("Converting beginning month date to end of month date")
            dt = dt + pd.tseries.offsets.MonthEnd()

        beta = 0.02
        cnst = 16
        wgt = max(0,np.tanh(beta*(loan_age - cnst))) #wgt for 15Y rate
        if pad:
            mtg = self.mtg_rates.ix[:dt,['15Y','30Y']].ix[-1]
        else:
            mtg = self.mtg_rates.ix[dt,['15Y','30Y']]
        return mtg.dot([wgt,1-wgt])

class Hist_Pool_Data(BBG_Data):
    """Parent for datastores of pool-specific data"""
    def __init__(self, path, read_only, p_root, field, data_name):
        """read_only must be False as we will be fetching data as we go."""
        super(Hist_Pool_Data, self).__init__(path=path, read_only=read_only)
        self.p_root = p_root
        self.field = field
        self.data_name = data_name

    def get_all_data(self):
        """We won't fetch stuff ahead of time for this one."""
        pass

    def get_cpr_for_cusip(self, cusip, fix_lag=True):
        """BBG Mortgage data is specified a month forward of the GNM files'
        'as_of_date'. Probably b/c it's really as-of the middle of the
        specified month.
        
        Calling fix_lag will put it on the same footing as the GNM data, i.e.,
        it'll still need to be lagged another month for prediction."""
        #Check if we already have it
        cusip = "_" + cusip
        address = self.p_root + cusip[3:5]
        toc = self.hdfstore.get_node(address)
        address = address + "/" + cusip

        if toc is not None and cusip in toc:
            data = self.hdfstore[address]
            if data.name == 'blank':
                logger.error("No {} data for {}".format(self.data_name, cusip[1:]))
                data = None
            elif data.index.dtype == 'int64': #legacy conversion
                logger.info("Legacy conversion of stored {} data index".format(self.data_name))
                data.index = [datetime.fromordinal(x).date() for x in
                              data.index]
                self.hdfstore[address] = data
        else:
            data = self.bbg.get_hist_pool_data(cusip[1:], field_code=self.field)
            if len(data) > 0:
                data.name = cusip
                self.hdfstore.append(address, data) 
            else: #No data available
                logger.error("No {} data for {}".format(self.data_name, cusip[1:]))
                data = pd.Series({0:0})
                data.name = 'blank'
                self.hdfstore.append(address, data) 
                data = None
            #put in a blank entry if there's no data.
            self.hdfstore.flush(fsync=True)

        if fix_lag and data is not None:
            data = data.shift(-1).dropna()
        return data

class CPR_Data(Hist_Pool_Data):
    """CPR for each pool"""
    def __init__(self, path= "data/cpr_data.h5", read_only=False):
        """read_only must be False as we will be fetching data as we go."""
        super(CPR_Data, self).__init__(path=path, read_only=read_only,
                                       p_root="cpr/", data_name="CPR",
                                      field="MTG_HIST_COLLAT_CPR_1MO")

class WAC_Data(Hist_Pool_Data):
    """Historical WAC for each pool"""
    def __init__(self, path= "data/wac_data.h5", read_only=False):
        """read_only must be False as we will be fetching data as we go."""
        super(CPR_Data, self).__init__(path=path, read_only=read_only,
                                       p_root="wac/", data_name="WAC",
                                      field="MTG_HIST_CPN")
