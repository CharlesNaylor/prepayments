# Collection of model objects. Gathers appropriate data. fits model.
from prepayments.bloomberg_data import Mortgage_Rate_Data, HPA_Data, YC_Data, CPR_Data
from prepayments.gnm_data import GNM_Pool, GNM_II_LL
import pandas as pd
from pandas.tseries.offsets import MonthEnd, MonthBegin, Day
import datetime
from time import mktime
import logging
import re
import numpy as np

logger = logging.getLogger()
class Model(object):
    """Parent container for prepayment models"""
    def __init__(self, scales, stan_path, readonly_yc=True):
        self.scales = scales
        self.stan_path = stan_path

        self.data_mtg_rates = Mortgage_Rate_Data()
        self.data_hpa = HPA_Data()
        self.data_yc = YC_Data(read_only=readonly_yc)

    def _compile_model(self):
        import pystan
        self.stanmodel = pystan.StanModel(self.stan_path)

    def fit(self, p_data):
        """Fit stan model"""
        raise NotImplementedException

class PoolModel(Model):
    """Container for modelling prepayments based on GNM Pool data"""
    SCALAR = pd.Series({'burnout':1e-6, 'cato':0.1, 'next_month_cpr':0.01, 'hpa':1,
                        'incentive':5e-5, 'lockin':1, 'sato':1e-4, 'seasonality':1,
                        'upfront_mip':0.01})

    def __init__(self, data_path=None, scales=None, stan_path=None,
                 readonly_yc=True):
        self.data_cpr = CPR_Data()
        self.data_pool = GNM_Pool(path=data_path)

        if scales is None:
            scales = self.SCALAR
        self.scales = scales

        if stan_path is None:
            stan_path = "stan/pool_poissonhurdle.stan"
        self.stan_path = stan_path
        super(PoolModel, self).__init__(scales, stan_path, readonly_yc)

    def current_month_data_for_cusip(self, cusip, dt=None):
        """Get data for the current month for forecasting"""

        if dt is None:
            dt = pd.Timestamp(pd.Timestamp.now().date())
        begin_dt = dt - MonthBegin()

        pool = self.data_pool.df_for_cusip(cusip)
        if len(pool) == 0:
            logger.error("Pools table has no records for {}, {:%Y-%m-%d}".format(cusip, dt))
            return None

        pool.index = pd.DatetimeIndex(pool.index)
        last_data_dt = pool.index[-1]
        prev_dt = pool.index[-2]
        pool = pool.ix[last_data_dt]

        if pool['pool_upb'] == 0:
            error_str = "Called current_month_data method for {}, but it was paid off last month."
            logging.error(error_str.format(cusip))
            return None

        if last_data_dt == begin_dt:
            warn_str = "Called current_month_data method for {}, but we have data for the current month."
            logging.warn(warn_str.format(cusip))
            return self.data_for_cusip(cusip, dt=begin_dt)
        elif last_data_dt < dt - MonthBegin(2) - Day():
            error_str = "Called current_month_data on {}, but the last data we have is {}, more than a month before."
            logging.error(error_str.format(cusip, last_data_dt))
            return None

        where_str = "(cusip == '{}') & (as_of_date == {})"
        where_str = where_str.format(cusip, mktime(last_data_dt.timetuple()))#, mktime(next_dt.timetuple()))
        msas, mip, various, states = [self.data_pool.msas_table.read_where(where_str),
                                    self.data_pool.insurancepremiums_table.read_where(where_str),
                                    self.data_pool.variouses_table.read_where(where_str),
                                    self.data_pool.states_table.read_where(where_str)]

        #Validate some basic data
        #A lot of these 99999s may indicate the loan was paid off this month.

        wac = pool['wac']
        wala = pool['wala']

        if wac == 99999:
            logger.error("Weighted Average Coupon is missing for {}, {:%Y-%m-%d}".format(cusip, dt))
            return None
        #Get loan origination date, and correct the wala.
        if wala == 999: #The loan might have been paid off this month.
            logger.error("Weighted avg loan age is missing for {}, {:%Y-%m-%d}".format(cusip, dt)) #Maybe I can find it another way?
            return None
        else:
            origination_dt = pool['as_of_date'] - MonthBegin(wala)
            wala = len(pd.date_range(origination_dt,dt,freq=MonthBegin()))

        #MIP doesn't necessarily exist.
        # MIP-related
        ## f(upfront MIP, WALA) (upfront MIP will be refunded on amortized basis)
        ## f(HPA) (extra incentive to refinance to end monthly MIP)

        if len(mip) == 0:
            logger.info("No mortgage insurance on {}".format(cusip))
            upfront_mip = 0
        else:
            mip = mip[0]
            upfront_mip_cols = [(x.group(0), int(x.group(1))/ 100.) for nm in
                                mip.dtype.names for x in
                                [re.match("upfront_mip_([0-9]{3})_pct_upb",nm)]
                                if x] #Fun! get colnames and the rate
            #True up mip rate % in case there's a lot of missing data
            mip_pct_sum = np.sum([v[1] for v in upfront_mip_cols])
            upfront_mip = np.sum([(v[1]/mip_pct_sum) * (mip[v[0]] / 100.) for v in upfront_mip_cols])

        #OTHER DATA
        #CATO: difference between curve now and curve at origination
        origination_yc = self.data_yc.yc.ix[origination_dt + MonthEnd(),
                                                        ['2Y', '10Y']].diff().dropna()
        new_yc = self.data_yc.fetch_dt(dt).ix[['2Y','10Y']].diff().dropna()
        cato = new_yc - origination_yc
        cato = cato.values[0]

        #SATO is unavailable for this one, since we can't calculate mean wacs before 2012.
        #We can use the prevailing benchmark mortgage rate, however.
        sato = (1e-3 * wac) - self.data_mtg_rates.get_rate_for_loan_age(0, origination_dt)

        #Seasonality
        seasonality = dt.month

        #Home Price Appreciation: try to get MSAs, fall back to states
        if len(msas) == 0:
            logger.error("No MSA data for {}, {:%Y-%m-%d}".format(cusip, dt))
            wgts = []
        else:
            msas = msas[0]
            wgts = pd.Series({'first_msa':msas['first_pct_upb'],
                              'second_msa':msas['second_pct_upb'],
                              'third_msa':msas['third_pct_upb']})
            wgts = wgts.where(wgts > 0).dropna() #Get rid of the NANs, which due to overflow will == -31073

        if len(wgts) == 0 or sum(wgts) < 0.75: #No msas or not enough MSA data
            logging.info("Not enough MSA data for {}. Trying state data".format(cusip))
            wgts = pd.Series(data=states["balance_pct"],index=states['state'])
            appreciation_sets = {nm:self.data_hpa.get_returns_for_state(nm) for nm in wgts.index}
        else:
            appreciation_sets = {nm:self.data_hpa.get_returns_for_msa_code(msas[nm]) for nm in wgts.index}

        appreciation_sets = pd.DataFrame({k:v for k,v in appreciation_sets.items() if v is not None})
        if len(appreciation_sets) == 0: #Couldn't find anything
            #go generic (national index)
            hpa = self.data_hpa.get_returns_for_msa_code(msa_code=None)
            hpa = hpa.ix[origination_dt:(dt+MonthEnd())].sum()
        else:
            wgts = wgts / wgts.sum() #normalize (renders /100. unnecessary, too)
            hpa = appreciation_sets.mul(wgts).sum(axis=1)
            hpa = hpa.ix[origination_dt:dt].sum() #If origination_dt < hpa.index[0], errors.

        #Incentive
        incentive = wac - (1000 *
                           self.data_mtg_rates.get_rate_for_loan_age(wala,
                                                                     last_data_dt))

        #Burnout: weighted cumulative incentive
        cum_incentive = 0
        for i in xrange(wala):
            cum_incentive += wac - (1000 * self.data_mtg_rates.get_rate_for_loan_age(i,
                                    origination_dt + MonthBegin(i), pad=True))

        # Turnover: Home Price Appreciation, "seasoning ramp", lock-in, seasonality
        ## Seasoning ramp: increasing function of WALA and HPA
        ## Lock-in: the effect of rising interest rates or negative equity. Current mtg rate / WAC. beta Must be < 1.0
        lockin = (1000 * self.data_mtg_rates.get_rate_for_loan_age(wala, last_data_dt)) / wac
        ## Seasonality: X-11/12 ARIMA per month. They say Census Bureau, so it's Demeter.

        ans = dict(dt=dt, cato=cato, sato=sato, seasonality=seasonality, hpa=hpa,
                   incentive=incentive, burnout=cum_incentive, lockin=lockin,
                   security_interest_rate=pool['security_interest_rate'],
                   wac=wac, upb=pool['pool_upb'], upfront_mip=upfront_mip, wala=wala, origination_dt=origination_dt)
        return ans


    def data_for_cusip(self, cusip, dt):
        """ * Curve at Origination:
                * pools.as_of_date, pools.pool_issue_date
                * monthly history of the yield curve. (where "Yield Curve" == 10Y-2Y spread)
            * Spread at Origination: median WAC of pool vs. average WAC of all pools in month
                * WAC
            * Seasonality: Expect higher prepayments in April-August (prefer summer move minus 1M lag)
                * as_of_date
            * Loan size
                * aols
            * Home Price Appreciation
                * first_msa, first_pct_upb, second_msa, second_pct_upb, etc.
                * MSA code concordance
                * CPI
            * Incentive: purchase mtg rate - function of 15Y and 30Y rates, WALA.
                * 15Y mtg rate, 30Y mtg rate, WALA
            * Burnout: weighted sum of monthly NPV of savings from refinancing
                * monthly incentives, w/ weights declining on accumulated home equity (home price appreciation + LTV)
            * Turnover: Home Price Appreciation, "seasoning ramp", lock-in, seasonality
                * Seasoning ramp: half-sigmoid function approaching 1 at ~ 11 months' WALA
                * Lock-in: the effect of rising interest rates or negative equity. Current mtg rate / WAC. Must be < 1.0
                * Seasonality: X-11/12 ARIMA per month. They say Census Bureau, so it's Demeter.
            * MIP-related
                * f(upfront MIP, WALA) (upfront MIP will be refunded on amortized basis)
                * f(HPA) (extra incentive to refinance to end monthly MIP)
            * CPR (this is the endo)
                * Wound up using the value from BBG, as while we have unpaid principal balance, and weighted average coupon, we don't have actual payment or scheduled payment.  """

        next_dt = dt + MonthBegin() #Year-month dates were stored as first of the month.
        prev_dt = dt - MonthBegin()
        if next_dt > pd.Timestamp.now():
            logger.error("Too soon to use data from {:%Y-%m-%d}".format(dt))
            return None

        where_str = "(cusip == '{}') & (as_of_date == {})"
        where_str = where_str.format(cusip, mktime(dt.timetuple()))#, mktime(next_dt.timetuple()))
        pool, msas, mip, various, states = [self.data_pool.pools_table.read_where(where_str),
                                    self.data_pool.msas_table.read_where(where_str),
                                    self.data_pool.insurancepremiums_table.read_where(where_str),
                                    self.data_pool.variouses_table.read_where(where_str),
                                    self.data_pool.states_table.read_where(where_str)]
        #mip handled differently b/c it may not exist

#        where_str = "(cusip == '{}') & (as_of_date >= {}) & (as_of_date <= {})"
#        where_str = where_str.format(cusip, mktime(dt.timetuple()), mktime(next_dt.timetuple()))
#        various = self.data_pool.variouses_table.read_where(where_str) 
#        if not len(various) == 2:
#            raise Excepton("Expected 2 records for CPR, got {}".format(len(various)))

        #Validate some basic data
        #A lot of these 99999s may indicate the loan was paid off this month.
        if len(pool) == 0:
            logger.error("Pools table has no records for {}, {:%Y-%m-%d}".format(cusip, dt))
            return None
        else:
            pool = pool[0]

        wac = pool['wac']
        wala = pool['wala']

        if any([wac == 99999, wala == 999]) and pool['pool_upb'] == 0:
            where_str = "(cusip == '{}') & (as_of_date == {})"
            old_data = self.data_pool.pools_table.read_where(where_str.format(cusip, mktime(prev_dt.timetuple())))
            if len(pool) > 0:
                wala = old_data[0]['wala'] + 1
                wac = old_data[0]['wac']

        if len(various) == 0:
            logger.info("Various table has no records for {}, {:%Y-%m-%d}".format(cusip, dt))
        else:
            various = various[0]
        if wac == 99999:
            logger.error("Weighted Average Coupon is missing for {}, {:%Y-%m-%d}".format(cusip, dt))
            return None
        #Get loan origination date
        if wala == 999: #The loan might have been paid off this month.
            logger.error("Weighted avg loan age is missing for {}, {:%Y-%m-%d}".format(cusip, dt)) #Maybe I can find it another way?
            return None
        else:
            if isinstance(pool['as_of_date'], (np.int32,int)):
                origination_dt = pd.Timestamp.fromtimestamp(pool['as_of_date']) - MonthBegin(wala)
            else:
                origination_dt = pool['as_of_date'] - MonthBegin(wala)

        # CPR (this is the endo)
        ## cf. docs with 'CPR' in the name. Due to lack of data on finished
        #pools in BBG, I'm using my calculations. But they're not the same as
        #BBG's.
        endo = self.get_endo_for_cusip(cusip, use_bbg=True)
        if endo is None:
            return None
        else:
            #burnout = endo.ix[:dt.date()].sum() #alternate burnout
            #endo = endo[next_dt.date()] #ENDO IS LAGGED!!!
            if next_dt in endo.index:
                endo = endo[next_dt]
            else:
                endo = pd.np.NaN
            next_month_cpr = 1 - (1-endo) ** 12
           #BBG CPR seems to be shifted forward already.
        #cusip = pool['cusip']

        #MIP doesn't necessarily exist.
        # MIP-related
        ## f(upfront MIP, WALA) (upfront MIP will be refunded on amortized basis)
        ## f(HPA) (extra incentive to refinance to end monthly MIP)

        if len(mip) == 0:
            logger.info("No mortgage insurance on {}".format(cusip))
            upfront_mip = 0
        else:
            mip = mip[0]
            upfront_mip_cols = [(x.group(0), int(x.group(1))/ 100.) for nm in
                                mip.dtype.names for x in
                                [re.match("upfront_mip_([0-9]{3})_pct_upb",nm)]
                                if x] #Fun! get colnames and the rate
            #True up mip rate % in case there's a lot of missing data
            mip_pct_sum = np.sum([v[1] for v in upfront_mip_cols])
            upfront_mip = np.sum([(v[1]/mip_pct_sum) * (mip[v[0]] / 100.) for v in upfront_mip_cols])

        #OTHER DATA
        #CATO: difference between curve now and curve at origination
        cato = self.data_yc.get_spread(dt + MonthEnd(), origination_dt + MonthEnd())

        #SATO: Median WAC of pool vs. Average median WAC of all pools per month
        # of origination
#        mean_wacs = pd.Series(data=self.data_pool.h5file.root.mean_wacs.values, 
#                              index=self.data_pool.h5file.root.mean_wacs.index)
#        mean_wacs.index = pd.to_datetime(mean_wacs.index)
#        sato = wac - mean_wacs.ix[dt + MonthEnd()]
        #SATO is unavailable for this one, since we can't calculate mean wacs before 2012.
        #We can use the prevailing benchmark mortgage rate, however.
        sato = (1e-3 * wac) - self.data_mtg_rates.get_rate_for_loan_age(0, origination_dt)

        #Seasonality
        seasonality = dt.month

        #Home Price Appreciation: try to get MSAs, fall back to states
        if len(msas) == 0:
            logger.error("No MSA data for {}, {:%Y-%m-%d}".format(cusip, dt))
            wgts = []
        else:
            msas = msas[0]
            wgts = pd.Series({'first_msa':msas['first_pct_upb'],
                              'second_msa':msas['second_pct_upb'],
                              'third_msa':msas['third_pct_upb']})
            wgts = wgts.where(wgts > 0).dropna() #Get rid of the NANs, which due to overflow will == -31073

        if len(wgts) == 0 or sum(wgts) < 0.75: #No msas or not enough MSA data
            logging.info("Not enough MSA data for {}. Trying state data".format(cusip))
            wgts = pd.Series(data=states["balance_pct"],index=states['state'])
            appreciation_sets = {nm:self.data_hpa.get_returns_for_state(nm) for nm in wgts.index}
        else:
            appreciation_sets = {nm:self.data_hpa.get_returns_for_msa_code(msas[nm]) for nm in wgts.index}

        appreciation_sets = pd.DataFrame({k:v for k,v in appreciation_sets.items() if v is not None})
        if len(appreciation_sets) == 0: #Couldn't find anything
            #go generic (national index)
            hpa = self.data_hpa.get_returns_for_msa_code(msa_code=None)
            hpa = hpa.ix[origination_dt:(dt+MonthEnd())].sum()
        else:
            wgts = wgts / wgts.sum() #normalize (renders /100. unnecessary, too)
            hpa = appreciation_sets.mul(wgts).sum(axis=1)
            hpa = hpa.ix[origination_dt:dt].sum() #If origination_dt < hpa.index[0], errors.

        #Incentive
        incentive = wac - (1000 *
                           self.data_mtg_rates.get_rate_for_loan_age(wala, dt))

        #Burnout: weighted cumulative incentive
        cum_incentive = 0
        for i in xrange(wala):
            cum_incentive += wac - (1000 * self.data_mtg_rates.get_rate_for_loan_age(i,
                                    origination_dt + MonthBegin(i)))

        # Turnover: Home Price Appreciation, "seasoning ramp", lock-in, seasonality
        ## Seasoning ramp: increasing function of WALA and HPA
        ## Lock-in: the effect of rising interest rates or negative equity. Current mtg rate / WAC. beta Must be < 1.0
        lockin = (1000 * self.data_mtg_rates.get_rate_for_loan_age(wala, dt)) / wac
        ## Seasonality: X-11/12 ARIMA per month. They say Census Bureau, so it's Demeter.

        ans = dict(cato=cato, sato=sato, seasonality=seasonality, hpa=hpa,
                   incentive=incentive, burnout=cum_incentive, lockin=lockin,
                   upfront_mip=upfront_mip, next_month_cpr=next_month_cpr,
                   next_month_smm=endo)
        return ans

    def get_samples_with_old_dates(self, n, gnm2_only=True,
                                    skew_early=True,
                                    pickle_path=None,
                                    csv_path=None):
        """Use BBG Cpr data we already have to get samples.
        
        :param n: number of samples to retrieve
        :type n: int
        :param gnm2_only: Only use GNM II pools
        :type gnm2_only: bool
        :param skew_early: Oversample early data to correct for the skew in the database towards newer pools
        :type skew_early: bool
        :param pickle_path: Where to save partial results in case there's a failure
        :type pickle_path: str
        :param csv_path: Where to save complete results (if at all)
        :type csv_path: str

        :rtype: pandas.Panel
        """
        import cPickle as pickle
        if pickle_path is None:
            pickle_path = "/data/prepayments/sample_data.pickle"
        if gnm2_only:
            sample_cusips = self.data_pool.h5file.root.candidate_pools.col("cusip")
            bad_cusip = "36230R6H4" #corrupted in the file and I can't figure out how to get rid of it.
            sample_cusips = np.delete(sample_cusips, np.where(sample_cusips ==
                                                              bad_cusip))
        else:
            with open("/media/gdaa/Charles/prepayment/sample_cusips.pickle","r") as f:
                sample_cusips = pickle.load(f)

        data = {}
        if skew_early:
            vintage_buckets = pd.Series(data=1,
                                        index=pd.date_range("1980-01-01","2017-01-01",
                                                            freq="12M"))
        try:
            for x in np.random.choice(sample_cusips,n,replace=False):
                new_data = self.data_for_cusip_random_dt(x,
                                                         skew_early=vintage_buckets)
                if new_data is not None:
                    if skew_early:
                        vintage = pd.Timestamp((new_data['dt'] -
                                               MonthBegin(new_data['wala'])).year,
                                                         1, 31)
                        vintage_buckets[vintage] += 1
                    data[(x,new_data['dt'])] = new_data
        finally:
            with open(pickle_path,"wb") as f:
                pickle.dump(data,f)

        data = pd.DataFrame.from_dict(data, orient='index')
        if csv_path is not None:
            data.reset_index().to_csv(csv_path)
        return data

    def get_endo_for_cusip(self, cusip, use_bbg=True):
        """For a given cusip, return the endogenous variable we will regress
        against.

        :param cusip: Cusip to retrieve
        :type cusip: str
        :rtype: pandas.Series"""
        if use_bbg:
            endo = self.data_cpr.get_cpr_for_cusip(cusip)
        else:
            endo = self.data_pool.smm_for_cusip(cusip)
        return endo

    def data_for_cusip_random_dt(self, cusip, skew_early=None, dt=None):
        """For a given cusip, figure out what dates we have CPR for,
        choose a random one, then retrieve data. Will heavily duplicate code in
        "self.data_for_cusip, so I'll have to figure out the best way to
        refactor at some point.

        Must use BBG CPR.

        :param cusip: cusip to look up
        :type cusip: str
        :param skew_early: if anything, should be a series giving the current
                               count of samples by of available dates by year.
                               If it's present, the method will sample skewed
                               to the least frequent years. Note this will
                               still leave a bias to more recent data, as the
                               vintages we are sampling from are biased recent.
        :type skew_early: pandas.Series
        :param dt: if anything, should be the date you want to retrieve
        (usually the current one)"""

        pool = self.data_pool.df_for_cusip(cusip)
        if len(pool) == 0:
            logger.error("Pools table has no records for {}".format(cusip))
            return None

        pool.index = pd.DatetimeIndex(pool.index)

        # CPR (this is the endo)
        ## cf. docs with 'CPR' in the name. Due to lack of data on finished
        #pools in BBG, I'm using my calculations. But they're not the same as
        #BBG's. 
        endo = self.get_endo_for_cusip(cusip, use_bbg=True)
        if endo is None:
            return None
        else:
            endo.index = pd.DatetimeIndex(endo.index)
            endo = endo.ix["1980-01-01":]
            if skew_early is not None: 
                preference = skew_early.ix[str(endo.index[0].year):str(endo.index[-1].year)]
                if len(preference) > 1:
                    preference = preference.pow(-1.)
                    preference = preference.div(preference.sum()).cumsum()
                    rnd = np.random.rand()
                    dt = pd.Timestamp(preference[preference >= rnd].index[0].year,
                                      np.random.randint(1,13),
                                      1)
                    if dt < endo.index[0] or dt > endo.index[-1]:
                        dt = pd.Timestamp(np.random.choice(endo.ix[str(dt.year)].index)) #Year should still be correct.
                else: #only one year available
                    dt = pd.Timestamp(np.random.choice(endo.index.copy()))
            else:
                dt = pd.Timestamp(np.random.choice(endo.index.copy()))
            endo = endo[dt]
        next_dt = dt + MonthBegin() #Year-month dates were stored as first of the month.
        prev_dt = dt - MonthBegin()
        if dt not in pd.DatetimeIndex(pool.index):
            tbl_dt = pool.index[0]
        else:
            tbl_dt = dt
        if next_dt > pd.Timestamp.now():
            logger.error("Too soon to use data from {:%Y-%m-%d}".format(dt))
            return None

        where_str = "(cusip == '{}') & (as_of_date == {})"
        where_str = where_str.format(cusip, mktime(tbl_dt.timetuple()))#, mktime(next_dt.timetuple()))
        msas, mip, states = [self.data_pool.msas_table.read_where(where_str),
                                    self.data_pool.insurancepremiums_table.read_where(where_str),
                                    self.data_pool.states_table.read_where(where_str)]
        #mip handled differently b/c it may not exist
        pool = pool.ix[tbl_dt]

        #Validate some basic data
        #A lot of these 99999s may indicate the loan was paid off this month.

        wac = pool['wac']
        wala = pool['wala']

        if any([wac == 99999, wala == 999]) and pool['pool_upb'] == 0:
            where_str = "(cusip == '{}') & (as_of_date == {})"
            old_data = self.data_pool.pools_table.read_where(where_str.format(cusip, mktime(prev_dt.timetuple())))
            if len(pool) > 0:
                wala = old_data[0]['wala'] + 1
                wac = old_data[0]['wac']

        if wac == 99999:
            logger.error("Weighted Average Coupon is missing for {}, {:%Y-%m-%d}".format(cusip, dt))
            return None
        #Get loan origination date, and correct the wala.
        if wala == 999: #The loan might have been paid off this month.
            logger.error("Weighted avg loan age is missing for {}, {:%Y-%m-%d}".format(cusip, dt)) #Maybe I can find it another way?
            return None
        else:
            origination_dt = pool['as_of_date'] - MonthBegin(wala)
            wala = len(pd.date_range(origination_dt,dt,freq=MonthBegin()))

        #MIP doesn't necessarily exist.
        # MIP-related
        ## f(upfront MIP, WALA) (upfront MIP will be refunded on amortized basis)
        ## f(HPA) (extra incentive to refinance to end monthly MIP)

        if len(mip) == 0:
            logger.info("No mortgage insurance on {}".format(cusip))
            upfront_mip = 0
        else:
            mip = mip[0]
            upfront_mip_cols = [(x.group(0), int(x.group(1))/ 100.) for nm in
                                mip.dtype.names for x in
                                [re.match("upfront_mip_([0-9]{3})_pct_upb",nm)]
                                if x] #Fun! get colnames and the rate
            #True up mip rate % in case there's a lot of missing data
            mip_pct_sum = np.sum([v[1] for v in upfront_mip_cols])
            upfront_mip = np.sum([(v[1]/mip_pct_sum) * (mip[v[0]] / 100.) for v in upfront_mip_cols])

        #OTHER DATA
        #CATO: difference between curve now and curve at origination
        cato = self.data_yc.get_spread(dt + MonthEnd(), origination_dt + MonthEnd())

        #SATO is unavailable for this one, since we can't calculate mean wacs before 2012.
        #We can use the prevailing benchmark mortgage rate, however.
        sato = (1e-3 * wac) - self.data_mtg_rates.get_rate_for_loan_age(0, origination_dt)

        #Seasonality
        seasonality = dt.month

        #Home Price Appreciation: try to get MSAs, fall back to states
        if len(msas) == 0:
            logger.error("No MSA data for {}, {:%Y-%m-%d}".format(cusip, dt))
            wgts = []
        else:
            msas = msas[0]
            wgts = pd.Series({'first_msa':msas['first_pct_upb'],
                              'second_msa':msas['second_pct_upb'],
                              'third_msa':msas['third_pct_upb']})
            wgts = wgts.where(wgts > 0).dropna() #Get rid of the NANs, which due to overflow will == -31073

        if len(wgts) == 0 or sum(wgts) < 0.75: #No msas or not enough MSA data
            logging.info("Not enough MSA data for {}. Trying state data".format(cusip))
            wgts = pd.Series(data=states["balance_pct"],index=states['state'])
            appreciation_sets = {nm:self.data_hpa.get_returns_for_state(nm) for nm in wgts.index}
        else:
            appreciation_sets = {nm:self.data_hpa.get_returns_for_msa_code(msas[nm]) for nm in wgts.index}

        appreciation_sets = pd.DataFrame({k:v for k,v in appreciation_sets.items() if v is not None})
        if len(appreciation_sets) == 0: #Couldn't find anything
            #go generic (national index)
            hpa = self.data_hpa.get_returns_for_msa_code(msa_code=None)
            hpa = hpa.ix[origination_dt:(dt+MonthEnd())].sum()
        else:
            wgts = wgts / wgts.sum() #normalize (renders /100. unnecessary, too)
            hpa = appreciation_sets.mul(wgts).sum(axis=1)
            hpa = hpa.ix[origination_dt:dt].sum() #If origination_dt < hpa.index[0], errors.

        #Incentive
        incentive = wac - (1000 *
                           self.data_mtg_rates.get_rate_for_loan_age(wala, dt))

        #Burnout: weighted cumulative incentive
        cum_incentive = 0
        for i in xrange(wala):
            cum_incentive += wac - (1000 * self.data_mtg_rates.get_rate_for_loan_age(i,
                                    origination_dt + MonthBegin(i)))

        # Turnover: Home Price Appreciation, "seasoning ramp", lock-in, seasonality
        ## Seasoning ramp: increasing function of WALA and HPA
        ## Lock-in: the effect of rising interest rates or negative equity. Current mtg rate / WAC. beta Must be < 1.0
        lockin = (1000 * self.data_mtg_rates.get_rate_for_loan_age(wala, dt)) / wac
        ## Seasonality: X-11/12 ARIMA per month. They say Census Bureau, so it's Demeter.

        ans = dict(dt=dt, cato=cato, sato=sato, seasonality=seasonality, hpa=hpa,
                   incentive=incentive, burnout=cum_incentive, lockin=lockin,
                   upfront_mip=upfront_mip, next_month_cpr=endo, wala=wala)
        return ans

class PopovaPoolModel(PoolModel):
    """Implements the model specified in Popova, Bayesian Forecasting of
    Prepayment Rates for Individual Pools of Mortgages.

    They use the Actual Payment (i.e. scheduled payment + prepayment) as their
    endo. This should be more stable and not zero-inflated."""

    def __init__(self, data_path=None, scales=None, stan_path=None):
        if stan_path is None:
            stan_path = "stan/pool_popova.stan"
        super(PopovaPoolModel, self).__init__(data_path, scales, stan_path)

    def get_endo_for_cusip(self, cusip, use_bbg=False):
        """Endo is ln(Actual Payment), which is just the difference of
        pool_upb."""
        if use_bbg:
            endo = super(PopovaPoolModel, self).get_endo_for_cusip(cusip,
                                                            use_bbg=True)
        else:
            endo = self.data_pool.df_for_cusip(cusip)['pool_upb']
            if len(endo) > 1:
                endo = endo.diff().dropna().mul(-1)
            else:
                logger.error("Not enough data to calculate payments for {}".format(cusip))
                endo = None
        return endo

    def data_for_cusip_random_dt(self, cusip):
        pool = self.data_pool.df_for_cusip(cusip)
        if len(pool) == 0:
            logger.error("Pools table has no records for {}".format(cusip))
            return None

        pool.index = pd.DatetimeIndex(pool.index)

        # CPR (this is the endo)
        ## cf. docs with 'CPR' in the name. Due to lack of data on finished
        #pools in BBG, I'm using my calculations. But they're not the same as
        #BBG's.
        endo = self.get_endo_for_cusip(cusip, use_bbg=False)
        if endo is None or endo.shape[0] < 2:
            return None
        else:
            endo.index = pd.DatetimeIndex(endo.index)
            endo = endo.ix["1980-01-01":"2016-12-31"]
            dt = pd.Timestamp(np.random.choice(endo.index[:-1].copy()))
            next_dt = dt + MonthBegin() #Year-month dates were stored as first of the month.
            endo = endo[next_dt]
        prev_dt = dt - MonthBegin()
        if dt not in pd.DatetimeIndex(pool.index):
            tbl_dt = pool.index[1]
            prev_dt = pool.index[0]
        else:
            tbl_dt = dt
            prev_dt = dt - MonthBegin()
        if next_dt > pd.Timestamp.now():
            logger.error("Too soon to use data from {:%Y-%m-%d}".format(dt))
            return None
        prior_balance = pool.ix[prev_dt]['pool_upb']
        pool = pool.ix[tbl_dt]
        wac = pool['wac']
        wala = pool['wala']
        warm = pool['warm']

        if any([wac == 99999, wala == 999, warm == 999]):
            if pool['pool_upb'] == 0:
                where_str = "(cusip == '{}') & (as_of_date == {})"
                old_data = self.data_pool.pools_table.read_where(where_str.format(cusip, mktime(prev_dt.timetuple())))
                if len(pool) > 0:
                    wala = old_data[0]['wala'] + 1
                    wac = old_data[0]['wac']
                    warm = old_data[0]['warm'] - 1
            else:
                logging.error("One of wac, wala, or warm is missing for {}, {:%Y-%m-%d}.".format(cusip, tbl_dt))
                return None

        mtg_rate = 1e-2*self.data_mtg_rates.get_rate_for_loan_age(wala, dt)
        wac = 1e-5*wac

        incentive = wac - mtg_rate
        spline = incentive ** 3

        #burnout needs the scheduled balance in absence of prepayments.
        # Balance_k = Balance_0 * \frac{(1 + \tfrac{c}{12})^n - (1+\tfrac{c}{12})^k}{(1+\tfrac{c}{12})^n-1}
        orig_bal = pool['original_aggregate_amount']
        n = warm + wala
        coupon = 1 + (pool['security_interest_rate'] * 1e-5)/12.

        scheduled_balance = orig_bal * (((coupon ** n)  - (coupon ** wala))/(coupon**n - 1))
        if scheduled_balance == 0:
            logging.error("Selected last month for {}, {:%Y-%m-%d}".format(cusip, tbl_dt))
            return None
        burnout = np.log(pool['pool_upb'] / scheduled_balance)

        seasonality = 1 if dt.month in [5,6,7,8] else 0

        yc = self.data_yc.yc.ix[dt+MonthEnd()].diff()[1]

        scheduled_payment = self.data_pool.scheduled_payment(prior_balance,
                                                             pool['security_interest_rate'],
                                                             wala,
                                                             wala+warm)

        return dict(cusip=cusip, dt=dt, endo=endo, incentive=incentive, spline=spline,
                    burnout=burnout, seasonality=seasonality, yc=yc, wala=wala, warm=warm, wac=wac,
                    original_balance=orig_bal, scheduled_payment=scheduled_payment,
                    mtg_rate=mtg_rate, coupon=coupon, log_endo=np.log(endo) if endo > 0 else np.NAN)

    def data_for_cusip(self, cusip, use_bbg_cpr=False):
        """Samples the entire available history for a given cusip"""

        endo = self.get_endo_for_cusip(cusip, use_bbg=use_bbg_cpr)
        if endo is None or endo.shape[0] < 2:
            return None
        endo.index = pd.DatetimeIndex(endo.index)
        endo = endo.ix[:'2016-12-31']
        forecast_endo = endo.shift(-1).dropna()

        pool = self.data_pool.df_for_cusip(cusip)
        if len(pool) == 0:
            logger.error("Pools table has no records for {}".format(cusip))
            return None
        pool.index = pd.DatetimeIndex(pool.index)

        first_pool = pool.ix[0,:] #static data will be sourced from here.
        #If we don't have the whole data history, we'll have to fill stuff in.
        if pool.index[0] > forecast_endo.index[0]:
            pool = pool.reindex(index=forecast_endo.index)
            pool['wac'].fillna(first_pool['wac'], inplace=True)
            pool['as_of_date'] = pool.index
            start_index = pool.index.tolist().index(pd.Timestamp(first_pool.as_of_date))
            for i in xrange(start_index-1,-1,-1):
                pool.ix[i,'wala'] = first_pool.wala - i
                pool.ix[i,'warm'] = first_pool.warm + i

        #Fix 99999s
        pool.ix[pool.wac == 99999,'wac'] = first_pool.wac

        #Iterate through iterative data demands
        mtg_rates = pd.Series(index=pool.index)
        for i in xrange(pool.shape[0]):
            if pool.ix[i,'wala'] == 999:
                pool.ix[i,'wala'] = pool.ix[i-1,'wala']+1
                pool.ix[i,'warm'] = pool.ix[i-1,'warm']-1

            mtg_rates.ix[i] = 1e-2*self.data_mtg_rates.get_rate_for_loan_age(pool.ix[i,'wala'], 
                                                                             pool.ix[i,'as_of_date'])
        incentive = pool.wac.mul(1e-5).sub(mtg_rates)
        spline = incentive.pow(3)

        #burnout needs the scheduled balance in absence of prepayments.
        # Balance_k = Balance_0 * \frac{(1 + \tfrac{c}{12})^n - (1+\tfrac{c}{12})^k}{(1+\tfrac{c}{12})^n-1}
        if use_bbg_cpr: #can't use burnout b/c we have no guarantee for pool_upb. just use loan age instead.
            burnout = pool.ix[:,'wala']
        else:
            orig_bal = pool['original_aggregate_amount']
            n = pool.ix[0,'warm'] + pool.ix[0,'wala']
            coupon = 1 + (pool['security_interest_rate'] * 1e-5)/12.
            coupon_n = pd.Series(index=pool.index, data=coupon**n)
            coupon_k = pd.Series({k:(coupon**v.wala) for k,v in pool.iterrows()})
            scheduled_balance = coupon_n.sub(coupon_k).div(coupon_n-1).mul(orig_bal)
            burnout = pd.Series(index=pool.index, data=[np.log(x) for x in
                                                        pool.pool_upb.div(scheduled_balance)])

        seasonality = pd.Series(index=pool.index, data=[1 if dt.month in
                                                        [5,6,7,8] else 0 for dt
                                                        in pool.index])

        yc = self.data_yc.yc
        yc = yc.reindex(index=yc.index-MonthBegin(), method='bfill')
        yc = yc['10Y'] - yc['2Y']
        yc = yc.ix[pool.index]

        ans = pd.DataFrame(dict(endo=forecast_endo, incentive=incentive,
                                spline=spline, burnout=burnout,wala=pool['wala'],
                                seasonality=seasonality,yc=yc))
        return ans

    def get_samples_with_old_dates(self, n, gnm2_only=True, pickle_path=None):
        """Use BBG Cpr data we already have to get samples."""
        import cPickle as pickle
        print "{} cusips. {}".format(n, pd.datetime.now())
        if pickle_path is None:
            pickle_path = "/data/prepayments/sample_data3.pickle"
        if gnm2_only:
            sample_cusips = self.data_pool.h5file.root.candidate_pools.col("cusip")
        else:
            with open("/media/gdaa/Charles/prepayment/sample_cusips.pickle","r") as f:
                sample_cusips = pickle.load(f)

        data = {}
        try:
            for x in np.random.choice(sample_cusips,n,replace=False):
                data[x] = self.data_for_cusip(x, use_bbg_cpr=True)
        finally:
            with open(pickle_path,"wb") as f:
                pickle.dump(data,f)
        print "Finished: {}".format(pd.datetime.now())
        return data

class VintagePoolModel(PopovaPoolModel):
    """Popova-based Pool Model with shrunk betas using pooled vintage"""

    def __init__(self, data_path=None, scales=None, stan_path=None):
        if stan_path is None:
            stan_path = "stan/vintage_pool.stan"
        super(VintagePoolModel, self).__init__(stan_path=stan_path)

    def data_for_cusip(self, cusip, use_bbg_cpr=False):
        """Samples the entire available history for a given cusip"""

        endo = self.get_endo_for_cusip(cusip, use_bbg=use_bbg_cpr)
        if endo is None or endo.shape[0] < 2:
            return None
        endo.index = pd.DatetimeIndex(endo.index)
        forecast_endo = endo.shift(-1).dropna()

        pool = self.data_pool.df_for_cusip(cusip)
        if len(pool) == 0:
            logger.error("Pools table has no records for {}".format(cusip))
            return None
        pool.index = pd.DatetimeIndex(pool.index)

        first_pool = pool.ix[0,:] #static data will be sourced from here.
        #If we don't have the whole data history, we'll have to fill stuff in.
        if pool.index[0] > forecast_endo.index[0]:
            pool = pool.reindex(index=forecast_endo.index)
            pool['wac'].fillna(first_pool['wac'], inplace=True)
            pool['as_of_date'] = pool.index
            start_index = pool.index.tolist().index(pd.Timestamp(first_pool.as_of_date))
            for i in xrange(start_index-1,-1,-1):
                pool.ix[i,'wala'] = first_pool.wala - i
                pool.ix[i,'warm'] = first_pool.warm + i

        #Fix 99999s
        pool.ix[pool.wac == 99999,'wac'] = first_pool.wac

        #Iterate through iterative data demands
        mtg_rates = pd.Series(index=pool.index)
        for i in xrange(pool.shape[0]):
            if pool.ix[i,'wala'] == 999:
                pool.ix[i,'wala'] = pool.ix[i-1,'wala']+1
                pool.ix[i,'warm'] = pool.ix[i-1,'warm']-1

            mtg_rates.ix[i] = 1e-2*self.data_mtg_rates.get_rate_for_loan_age(pool.ix[i,'wala'], 
                                                                             pool.ix[i,'as_of_date'])
        incentive = pool.wac.mul(1e-5).sub(mtg_rates)
        spline = incentive.pow(3)

        #burnout needs the scheduled balance in absence of prepayments.
        # Balance_k = Balance_0 * \frac{(1 + \tfrac{c}{12})^n - (1+\tfrac{c}{12})^k}{(1+\tfrac{c}{12})^n-1}
        if use_bbg_cpr: #can't use burnout b/c we have no guarantee for pool_upb. just use loan age instead.
            burnout = pool.ix[:,'wala']
        else:
            orig_bal = pool['original_aggregate_amount']
            n = pool.ix[0,'warm'] + pool.ix[0,'wala']
            coupon = 1 + (pool['security_interest_rate'] * 1e-5)/12.
            coupon_n = pd.Series(index=pool.index, data=coupon**n)
            coupon_k = pd.Series({k:(coupon**v.wala) for k,v in pool.iterrows()})
            scheduled_balance = coupon_n.sub(coupon_k).div(coupon_n-1).mul(orig_bal)
            burnout = pd.Series(index=pool.index, data=[np.log(x) for x in
                                                        pool.pool_upb.div(scheduled_balance)])

        seasonality = pd.Series(index=pool.index, data=[1 if dt.month in
                                                        [5,6,7,8] else 0 for dt
                                                        in pool.index])

        yc = self.data_yc.yc
        yc = yc.reindex(index=yc.index-MonthBegin(), method='bfill')
        yc = yc['10Y'] - yc['2Y']
        yc = yc.ix[pool.index]

        vintage = pool['pool_issue_date']

        ans = pd.DataFrame(dict(endo=forecast_endo, incentive=incentive,
                                spline=spline, burnout=burnout,wala=pool['wala'],
                                seasonality=seasonality,yc=yc, vintage=vintage))
        return ans

class LoanModel(Model):
    SCALAR = pd.Series()
    """Container for modelling prepayments based on GNM II Loan-level data"""
    def __init__(self, data_path=None, scales=None, stan_path=None):
        self.data_loan = GNM_II_LL(path=data_path)

        if scales is None:
            scales = self.SCALAR
        self.scales = scales

        if stan_path is None:
            stan_path = "stan/loan.stan"
        self.stan_path = stan_path
        super(LoanModel, self).__init__(scales, stan_path)


    def _cusip_for_pool_id(self, pool_id):
        """Return cusip for pool_id
        TODO: implement cache"""
        where_str = "pool_id == '{}'".format(pool_id)
        pool = self.data_loan.pools_table.read_where(where_str)
        if len(pool) == 0:
            logger.error("No pool found for id {}".format(pool_id))
            return None
        else:
            return pool[0]['cusip']

    def get_samples(self, n, pickle_path=None, csv_path=None):
        """Return nicely formatted samples, optionally saving to disk."""
        data = []
        sample_indexes = np.random.choice(self.data_loan.loans_table.shape[0], n)
        for i in sample_indexes:
            data.append(self.data_for_loan(*self.data_loan.get_id_for_index(i)))
        data = pd.DataFrame.from_records([x for x in data if x is not None])
        data.set_index(['dsn','as_of_date'], inplace=True)
        if pickle_path is not None:
            import cPickle as pickle
            with open(pickle_path, "wb") as f:
                pickle.dump(data, f)
        if csv_path is not None:
            data.reset_index().to_csv(csv_path, index=False)
        return data

    def data_for_loan(self, dsn, dt):
        """ Return formatted data for a given loan on a given date"""
        where_str = "(dsn == {}) & (as_of_date >= {})"
        if isinstance(dt, np.int32):
            where_str = where_str.format(dsn, dt)
            dt = pd.Timestamp.fromtimestamp(dt)
        else:
            where_str = where_str.format(dsn, mktime(dt.timetuple()))

        loan = self.data_loan.loans_table.read_where(where_str)
        #mip handled differently b/c it may not exist

        if len(loan) == 0:
            logger.error("Loans table has no records for {}, {:%Y-%m-%d}".format(dsn, dt))
            return None
        elif len(loan) == 1:
            error_str = "{}, {:%Y-%m-%d}".format(dsn, dt)
            logger.error("Loans table doesn't have records for the" + \
                         " following month for " + error_str + \
                         ", so I can't test prediction on it.")
            return None
        else:
            endo = loan[1]['current_month_liquidation']
            removal_reason = loan[1]['removal_reason']
            loan = loan[0]
            loan = pd.Series(dict(zip(loan.dtype.names,loan))).to_frame().T
            loan = self.data_loan.format_df(loan, self.data_loan.loans_table).ix[0]

        #loan_origination_dt is mostly blank
        loan_age = loan['loan_age']
        origination_dt = loan['as_of_date'] - MonthBegin(loan_age)
        annual_mip = loan['annual_mip']

        pool_cusip = self._cusip_for_pool_id(loan['pool_id'])
        rate = 1e-3 * loan['loan_interest_rate']

        #OTHER DATA
        #CATO: difference between curve now and curve at origination
        cato = self.data_yc.get_spread(dt + MonthEnd(), origination_dt + MonthEnd())

        sato = rate - self.data_mtg_rates.get_rate_for_loan_age(0, origination_dt)

        #Seasonality
        seasonality = dt.month

        #Home Price Appreciation: states only
        hpa = self.data_hpa.get_returns_for_state(loan['state'])
        if hpa is None:
            hpa = self.data_hpa.get_returns_for_msa_code(msa_code=None)
            hpa = hpa.ix[origination_dt:(dt+MonthEnd())].sum()
        else:
            hpa = hpa.ix[origination_dt:(dt+MonthEnd())].sum()

        #Incentive
        incentive = rate - self.data_mtg_rates.get_rate_for_loan_age(loan_age, origination_dt)

        #Burnout: weighted cumulative incentive
        cum_incentive = 0
        for i in xrange(loan_age):
            cum_incentive = rate - self.data_mtg_rates.get_rate_for_loan_age(i, origination_dt + MonthBegin(i))

        ## Lock-in: the effect of rising interest rates or negative equity. Current mtg rate / WAC. beta Must be < 1.0
        lockin = self.data_mtg_rates.get_rate_for_loan_age(loan_age, dt) / rate

        #Everything else, except stuff I already have.
        everything = [u'agency', u'buy_down_status', u'credit_score', u'down_payment_assistance', u'first_time', 
                      u'issuer_id', u'loan_gross_margin', u'loan_purpose', u'ltv', u'maturity_date', 
                      u'months_delinquent', u'months_prepaid', u'number_of_borrowers', u'original_loan_term', 
                      u'original_principal_balance', u'origination_type', u'property_type', u'refinance_type',
                      u'remaining_loan_term', u'seller_issuer_id', u'state', 
                      u'total_debt_expense_ratio', u'upb', u'upb_at_issuance', u'upfront_mip']

        ans = dict(pool_cusip=pool_cusip, dsn=dsn, as_of_date=dt, origination_date=origination_dt,
                   prepaid=endo, removal_reason=removal_reason, loan_age=loan_age, annual_mip=annual_mip,
                   cato=cato, sato=sato, seasonality=seasonality, hpa=hpa, incentive=incentive,
                   burnout=cum_incentive, lockin=lockin)
        ans.update(loan[everything])
        return ans
