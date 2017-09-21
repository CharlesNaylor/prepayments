# Prepayments Modeling Journal
Text-only attempt to keep track of what I'm doing on this.

## 2017-01-18

Steve got back to me that they're primarily interested in 30Y Ginnie Mae II MBS prepayment risk. Ginnie Mae II MBS can have varying coupon rates and multiple issuers. Both Ginnie Mae I and II tend to have more prepayment risk than Freddie Mac pools b/c they are expeected to have high LTV and mandatory mortgage insurance which make refinancing attractive.

Ginnie Mae started giving loan-level data only in late 2013. Before then, we have pool-level data. I'll need a way to combine the two. I could also consider using Freddie Mac loan-level data for the earlier period, and comparing FM to GM since 2013.

I also need home price appreciation data. Steve recommended a set of state-level median home prices data. He was talking about series with tickers like `MTMHOHIO Index`, but I think *Existing* home prices, `MEHPOHIO Index` would be more appropriate. Home price appreciation affects LTV, and hence propensity to refinance, especially when you have mortgage insurance (i.e. LTV falls below 80%).

### To do on the data:
    -0. Get the GM loan level data as far back as possible-
    1. Get the GM pool data as far back as possible
      a. Get the Freddie Mac loan level data?
    2. Get the Home price appreciation data
    3. Write parsers for the data

### The model
    http://www.beyondbond.com/pdf/2.pdf and http://www.investinginbonds.com/assets/files/JP_Morgan_Prepayment_Model.pdf seem to be multiplicative models. The first link calls it's model 'multiplicative splines'. In the first link, CPR is regressed against 7 factors, each of which has some kind of polynomial fit applied to it. I guess they say they're multiplying them b/c they mention they actually want CPR in log space. So we're back to standard additive model design in log space.
    I don't really have much more to say on this until I have the data nicely formatted and I can look at some graphs.

## 2017-01-23

Back from the Stan conference. I have retrieved a lot of data that seems like it might be relevant. I have:
  * GNM II loan level data monthly back to 2013-10.
  * GNM Pool data monthly back to 2012-02.
  * Freddie Mac Loan Level data quarterly back to 1999.
  * Fannie Mae loan level (I think?) quarterly back to 2000.

Better start with the GNM II data. This is fixed-width-format, which won't be fun. I have monthly text files with pool-level and loan-level info (see <a href=http://www.ginniemae.gov/doing_business_with_ginniemae/investor_resources/mbs_disclosure_data/Lists/LayoutsAndSamples/Attachments/170/llmon2_layout.pdf> here </a>). I've been looking for alternatives, but I think my best bet is unfortunately going to be h5 files, as they compress well, don't need to be kept in-memory, and are compatible with both R and Python (in theory). I hated these things last time I used them, especially b/c they had a tendency to be corrupted.
The file layout pdf I linked above says we are on v. 1.6 of the format for these things, but doesn't give any info on previous formats. Hopefully they've updated the old files to the new format. I think I'm going to need to parse these line-by-line, rather than using pandas.read_fwf, as there are internal header files delineating each pool.

I hadn't planned on modelling per-pool effects, so I'm not sure if I should preserve that structure. It'll be easier to parse that way, though.So, I think the h5 file hierarchy is going to mimic the text files, at least for now. YearMon -> Pool -> Loan. This makes some sense, as in future we'll be updating by monthly files.

## 2017-01-24

Settled on an h5file format. Just 3 tables at root, one for Pools, one for Loans, one for Files. Loans already contain pointers to pools, and files will make sure we don't add something twice. Really, what I want to see are samples of the Loans table. There may be pool-level effects, but I can index those from Loans.

Now I'm adding everything in. This will take a while. The file's going to be upwards of 10GB.

While that's running, I'm also going to need the per-state house price data, and possibly per-metropolitan area, if it's available.

## 2017-01-25

Each loan-level file takes about half an hour to process, but the newer files are bigger, so I'm still waiting for it to finish. Should be done by 2PM or so. Meanwhile, the State-level house price data Steve suggested only seems to go back to 2010. It'd be better if we could get back further. Case-Schiller does the larger Metropolitan Statistical Areas back to 2000; it may be that a majority of the data can use those. GNM II gives 5-digit MSA codes, but these were apparently revised in 2015. Did GNM revise their codes? How different are the Census' codes before and after? 2015 is <a href=https://www.census.gov/population/metro/data/def.html>here</a>, and the older, 2006 codes are <a href=https://www.census.gov/population/metro/data/defhist.html>here</a>. Specifically, I'm using <a href=https://www.census.gov/population/metro/files/lists/2015/List1.xls>2015</a> and <a href=https://www.census.gov/population/metro/files/lists/2006/List1.txt>2006</a>.

**16:43:34**: Having finished all the data loading, it seems we *don't* have MSA-level data. The field is there but is blank in every case. So, on the state-level stuff, I could backfill with MSA data for some states. I.e., assume NY state before 2010 is adequately represented by the NYC metropolitan area. A bit risky. The other option is basic Case-Shiller. Maybe I can find some other source of state-level data.

**17:26:55** Should I just start looking at pool-level data, to start with, anyway? I only see it on the GNM website back to 2012, but the beyondbond pdf says they're using back to 1984. I'll have to check the Glenn Shultz textbook, see if he says where to get the older data.

## 2017-01-26

Shultz seems to use only a single month of loan-level data. I'm going to have to redo my (7GB, in the end) database because I was too stingy with some value sizes. Eg. loan_age got an 8-bit integer, but needed a 16-bit. I could take the opportunity to remove some fields that are always blank, like 'msa', and probably shrink the file down quite a bit. So what else is blank?

* MSA
* loan_origination_date is blank before 2015-04
* combined_LTV

That's it, amazingly. I'll take out MSA and combined_LTV, but I doubt it'll make it much smaller. I'm going to start this thing up again, and maybe go work on mortgage pools on the other computer while it's running.

## 2017-01-30

Got all the GNM pool data into a separate h5 file over the weekend. The loan-level data should be all fixed now, but I probably need to spot-check the pool-level data.

I think first I'm going to work through the JPM paper, which is pool-level data.  The Mortgages w/R book example uses loan-level data, which I'd prefer to switch to, but as a first deliverable, better to stick with what was directly requested. The pool-level data has significantly more structure to it, with 9 tables to the loan-level's 3. 

...and, there's a problem with the pool-level data. Unlike loan-level, they actually realigned a bunch of data in 2014, meaning I need to re-run the data gatherer. So, while that's going, back to loan-level, I guess.

## 2017-01-31

Here's my list of factors w/necessary fields for the JPM pool-level model:
    * Curve at Origination: 
        * pools.as_of_date, pools.pool_issue_date
        * monthly history of the yield curve. (where "Yield Curve" == 10Y-2Y spread)
    * Spread at Origination: median WAC of pool vs. average WAC of all pools in month
        * WAC
    * Seasonality: Expect higher prepayments in April-August (prefer summer move minus 1M lag)
        * as_of_date
    * Loan size
        * AOLS, plus quartiles. Anything useful to be done with quartiles? A
          generative model could use them.
    * Home Price Appreciation
        * first_msa, first_pct_upb, second_msa, second_pct_upb, etc.
        * MSA code concordance
        * CPI
    * Burnout: weighted sum of monthly NPV of savings from refinancing
        * monthly incentives, w/ weights declining on accumulated home equity (home price appreciation + LTV)
    * Turnover: Home Price Appreciation, "seasoning ramp", lock-in, seasonality
        * Seasoning ramp: half-sigmoid function approaching 1 at ~ 11 months' WALA
        * Lock-in: the effect of rising interest rates or negative equity. Current mtg rate / WAC. Must be < 1.0
        * Seasonality: X-11/12 ARIMA per month. They say Census Bureau, so it's Demeter.
    * Incentive: purchase mtg rate - function of 15Y and 30Y rates, WALA.
        * 15Y mtg rate, 30Y mtg rate, WALA
    * MIP-related
        * f(upfront MIP, WALA) (upfront MIP will be refunded on amortized basis)
        * f(HPA) (extra incentive to refinance to end monthly MIP)
    * CPR (this is the endo)
        * % of UPB of loans paid off (in Various table) +, I guess, repurchased, foreclosed, etc., in same table.

On the pool-level model, we *do* have MSA data.

## 2017-02-01

Set up datastores for bloomberg data. I made a concordance for MSA codes and Case-Schiller city indexes, or, failing those, CoreLogic state indexes. Now I'm doing some preprocessing, specifically pre-calculating all the average WACs per origination month, for SATO. It looks like there'll be more cleaning to do, as GNM has chosen to use 99999 to denote missing data, and I need to be on the look out for it. Wonder how common missing data is?

## 2017-02-03

CATO and SATO data done. cf. <a href="20170130%20-%20Examining%20GNMA%20II%20Pool-Level%20Data.ipynb"> Pool-level data notebook</a> for more details on each factor above. I'm making a "factors for pool number" method in `models.py`.

It's pretty clear that I'll need to pay a lot of attention to missing data all through this. The question is, can I just skip incomplete records, or are they somehow different from complete ones? If I do the proper Bayesian generative model, I'll just model them directly.

## 2017-02-06

Finished up data gathering for a single loan. I think I need to write a test suite, though.

## 2017-02-07

Debugging the data gathering. Got all the way through, but there's a problem with the collection of CPR that will mean I need to recreate the database from scratch (left off a page of records from the Various table).

While that's going on, I can start looking at the distributions of the rest of the data.

## 2017-02-08

How does % of UPB convert to SMM or CPR? Tricky. I don't seem to be able to get CPR from Bloomberg in bulk as an alternate.

## 2017-02-09

I wound up using UPB of loans paid off / original_loan_amount. Hope that's ok.  Did a bunch of other bugfixes. I need to make sure I have all missing MSAs accounted for in the concordance. I see I have 63 missing that are in 'first_msa' in the table, so I'll have to automate adding them.

## 2017-02-10

I'm not sure this UPB is really equivalent. I expanded the Bloomberg object to fetch CPRs. I can only get those for on-the-run MBSs, though. I still need to find equivalence from the GNM data.
Hopefully I don't go over the data allowance. Also numerous bugfixes to model.py, mostly handling missing data. Also direct fractional state handling for when MSA is inadequate.

## 2017-02-13

It seems 92% of CUSIPs I tested have BBG CPR data, so I'm going to stick with that.

## 2017-02-14

Everything squared away. Distributions checked, scalars picked. Tomorrow will start on the model.

## 2017-02-15

I should try to implement one of the frequentist spline models, for comparison if nothing else. BondLab is an obvious choice. My real goal would be a Bayesian mixture model, esp. if Steve wants a white paper on it.

## 2017-02-20

Started writing the model, but noticed that in my set of 10,000 samples, not one record came from March - May. There must be consistent missing data in those months. I'll have to find out what it is.

## 2017-02-21

Nevermind on missing month data. Seems to have been an artifact of looking at graphs over remote desktop.

## 2017-02-28

Notes have mostly moved over to R and ipynb notebooks in this doc directory.  Having trouble, generally, with getting much of regressive power on CPR data.  Maybe this is because I'm using BBG CPR data, and Bloomberg doesn't keep it for paid-off pools. So my data set consists entirely of not-fully-paid-off pools. I think I need to work harder on extracting CPR directly from the GNM data.
