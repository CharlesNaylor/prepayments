'''
.. module:: datasources
   :synopsis: A class for handling the Bloomberg API, plus a method for getting series from Burrito.
.. moduleauthor:: Charles Naylor <charles.naylor@nikkoam.com>

A class for handling the Bloomberg API, plus a method for getting series from Burrito.'''
import datetime
import logging
import pandas as pd
import blpapi


class Datasource(object):
    """
    General datasource for filling Series data
    """

    def __init__(self):
        pass

    def connect(self):
        raise NotImplementedError()

    def get_history(self, start_date=None, end_date=None):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

class Bloomberg(Datasource):
    '''Wrapper for blpapi'''
    # default_params = {'periodicityAdjusment':'ACTUAL'
    # , 'periodicitySelection':'DAILY'
    #                       , 'maxDataPoints':100000}

    default_params = {'periodicitySelection': 'MONTHLY'
        , 'maxDataPoints': 100000, 'periodicityAdjustment': "CALENDAR"}
    BBG_DATE_FMT = '%Y%m%d'
    BBG_ADDRESS = {'host': 'localhost', 'port': 8194}

    def __init__(self):
        super(Bloomberg, self).__init__()
        self.connect()

    def connect(self):

        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost(Bloomberg.BBG_ADDRESS['host'])
        sessionOptions.setServerPort(Bloomberg.BBG_ADDRESS['port'])

        logging.info("Connecting to %s:%s" % (Bloomberg.BBG_ADDRESS['host'], Bloomberg.BBG_ADDRESS['port']))
        # Create a Session
        session = blpapi.Session(sessionOptions)

        # Start a Session
        if not session.start():
            logging.error("Failed to start Bloomberg session.")

        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            logging.error("Failed to open //blp/refdata")

        # Obtain previously opened service
        self.service = session.getService("//blp/refdata")  # Start a Session

        self.session = session

    def get_history(self, securities, fields="PX_LAST"
                    , start_date=datetime.date(2000, 1, 1), end_date=datetime.date.today()
                    , other_params=default_params, tolerate_missing_data=False):
        '''Request historical data.

        :param securities: BBG tickers
        :type securities: string or list
        :param fields: BBG fields
        :type fields: string or list
        :param start_date: Date from which to retrieve data
        :type start_date: datetime.date
        :param end_date: Date to which to retrieve data
        :type end_date: datetime.date
        :param other_params: dictionary with parameterName:value
        :type other_params: dictionary
        :param tolerate_missing_data: Fill in missing data with None (especially useful for unusual field requests)
        :type tolerate_missing_data: boolean

        :rtype: pandas Series, DataFrame, or Panel, depending on the number of securities and fields provided.
        '''

        # Validate parameters. Due to python's default string indexing, it's necessary to ensure you have a list, not a string for these.
        if isinstance(securities, (str, unicode)):
            securities = [securities]
        if isinstance(fields, (str, unicode)):
            fields = [fields]

        request = self.service.createRequest("HistoricalDataRequest")

        for security in securities:
            request.getElement("securities").appendValue(security)

        for field in fields:
            request.getElement("fields").appendValue(field)

        request.set("startDate", start_date.strftime(Bloomberg.BBG_DATE_FMT))
        request.set("endDate", end_date.strftime(Bloomberg.BBG_DATE_FMT))

        for param, value in other_params.iteritems():
            request.set(param, value)

        logging.info(
            "Requesting {fields} for {securities}.".format(fields=", ".join(fields), securities=", ".join(securities)))

        if tolerate_missing_data:
            def custom_get_float(p_element, field):
                try:
                    ans = p_element.getElementAsFloat(field)
                except blpapi.NotFoundException:
                    ans = None
                return ans
        else:
            def custom_get_float(p_element, field):
                return p_element.getElementAsFloat(field)

        self.session.sendRequest(request)
        results = {}
        while (True):
            event = self.session.nextEvent(500)
            if event.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE):
                for msg in event:
                    element = msg.getElement("securityData")
                    security_name = element.getElementAsString("security")
                    field_data = element.getElement("fieldData")
                    data_length = field_data.numValues()
                    values = {}
                    if len(fields) > 1:
                        for i in range(data_length):
                            temp_val = field_data.getValueAsElement(i)
                            values[temp_val.getElementAsDatetime('date')] = pd.Series(
                                [custom_get_float(temp_val, field) for field in fields], index=fields)
                        results[security_name] = pd.DataFrame(values).T
                    else:
                        for i in range(data_length):
                            temp_val = field_data.getValueAsElement(i)
                            values[temp_val.getElementAsDatetime('date')] = custom_get_float(temp_val, fields[0])
                        results[security_name] = pd.Series(values)
            if event.eventType() == blpapi.Event.RESPONSE:
                break

        # Need to set indexes properly, or you'll just get a big ndArray of datetime.dates.
        # todo: multiple fields and 1 security should return a DataFrame, not a Panel
        if len(fields) == 1:
            pandas_results = pd.DataFrame(results)
            pandas_results.index = pd.DatetimeIndex(pandas_results.index)
            if len(securities) == 1:  # Extract the series
                pandas_results = pandas_results[pandas_results.columns[0]]
        else:
            pandas_results = pd.Panel(results)
            pandas_results.major_axis = pd.DatetimeIndex(pandas_results.major_axis)

        return pandas_results

    def get_reference_datum(self, security, field_code):
        """
        Return a reference datum for a security
        :param security: security for which to fetch the reference data
        :type security: str
        :param field_code: BBG field code
        :type field_code: str

        :return: The reference datum
        :rtype: str
        """

        request = self.service.createRequest("ReferenceDataRequest")
        request.getElement("securities").appendValue(security)
        request.getElement("fields").appendValue(field_code)

        logging.info("Requesting {1} for {0}".format(security, field_code))

        self.session.sendRequest(request)

        while (True):
            event = self.session.nextEvent(500)
            if event.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE):
                for msg in event:
                    element = msg.getElement("securityData").getValue(0)
                    security_name = element.getElementAsString("security")
                    field_datum = element.getElement("fieldData")
                    if field_datum.numValues() == 0:
                        raise Exception("BBG request for {0} for {1} returned nothing.".format(field_code, security))
                    else:
                        val = field_datum.getElement(0).getValue(0)
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        return val

    def get_reference_data(self, security, field_code):
        """
        Return a set of reference data for a security
        :param security: security for which to fetch the reference data
        :type security: str
        :param field_code: BBG field code
        :type field_code: str

        :return: The reference data as strings
        :rtype: list
        """

        request = self.service.createRequest("ReferenceDataRequest")
        request.getElement("securities").appendValue(security)
        request.getElement("fields").appendValue(field_code)

        logging.info("Requesting {1} for {0}".format(security, field_code))

        self.session.sendRequest(request)

        val = []
        while (True):
            event = self.session.nextEvent(500)
            if event.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE):
                for msg in event:
                    element = msg.getElement("securityData").getValue(0)
                    security_name = element.getElementAsString("security")
                    field_datum = element.getElement("fieldData")
                    if field_datum.numValues == 0:
                        raise Exception("BBG request for {0} for {1} returned nothing.".format(field_code, security))
                    else:
                        for i in range(field_datum.getElement(0).numValues()):
                            val.append(field_datum.getElement(0).getValue(i).getElement(0).getValue(0))
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        return val

    def get_hist_pool_data(self, cusip, field_code="MTG_HIST_COLLAT_CPR_1MO"):
        request = self.service.createRequest("ReferenceDataRequest")
        request.getElement("fields").appendValue(field_code)

        logging.info("Requesting {1} for {0}".format(cusip, field_code))

        if isinstance(cusip, list):
            for cusip_req in cusip:
                request.getElement("securities").appendValue("/cusip/{}".format(cusip_req))
        else:
                request.getElement("securities").appendValue("/cusip/{}".format(cusip))
        self.session.sendRequest(request)
        results = {}
        while (True):
            event = self.session.nextEvent(500)
            if event.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE):
                for msg in event:
                    element = msg.getElement("securityData").getValue(0)
                    security_name = element.getElementAsString("security")
                    field_data = element.getElement("fieldData")
                    if not field_data.isNull():
                        field_data = element.getElement("fieldData").getElement(0)
                        data_length = field_data.numValues()
                        values = {}
                        for i in range(data_length):
                            temp_dt = field_data.getValueAsElement(i).getElement(0)
                            temp_val = field_data.getValueAsElement(i).getElement(1)
                            values[temp_dt.getValue()] = temp_val.getValue()
                        values = pd.Series(values)
                        values.name = cusip
                        values.index = pd.DatetimeIndex(values.index)
                        results[cusip] = values
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        if not isinstance(cusip, list) and len(results) > 0:
                results = results[cusip]
        return results

    def get_holidays(self, calendar_code=None, start_date=None, end_date=None):
        """
        Return a set of holidays for a given security and date range

        :param securities:
        :param start_date:
        :param end_date:
        :return: securities with lists of holiday dates
        :rtype: dict
        """

        if start_date is None:
            start_date = datetime.date(2000, 1, 1)
        if end_date is None:
            end_date = datetime.date.today()

        request = self.service.createRequest("ReferenceDataRequest")

        fields = ["CALENDAR_HOLIDAYS"]
        # fields = ["CALENDAR_NON_SETTLEMENT_DATES"]

        request.getElement("securities").appendValue("SPX Index")

        for field in fields:
            request.getElement("fields").appendValue(field)

        overridefields = request.getElement("overrides")
        overrides = request.getElement("overrides")
        override1 = overrides.appendElement()
        override1.setElement("fieldId", "SETTLEMENT_CALENDAR_CODE")
        override1.setElement("value", calendar_code)
        override2 = overrides.appendElement()
        override2.setElement("fieldId", "CALENDAR_START_DATE")
        override2.setElement("value", start_date.strftime(Bloomberg.BBG_DATE_FMT))
        override3 = overrides.appendElement()
        override3.setElement("fieldId", "CALENDAR_END_DATE")
        override3.setElement("value", end_date.strftime(Bloomberg.BBG_DATE_FMT))

        logging.info("Requesting holidays for {0} between {1:%Y-%m-%d} and {2:%Y-%m-%d}".format(calendar_code
                                                                                                , start_date, end_date))

        self.session.sendRequest(request)
        results = {}
        while (True):
            event = self.session.nextEvent(500)
            if event.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE):
                for msg in event:
                    element = msg.getElement("securityData").getValue(0)
                    security_name = element.getElementAsString("security")
                    field_data = element.getElement("fieldData")
                    if not field_data.isNull():
                        field_data = element.getElement("fieldData").getElement(0)
                        data_length = field_data.numValues()
                        values = []
                        for i in range(data_length):
                            temp_val = field_data.getValueAsElement(i).getElement(0)
                            values.append(temp_val.getValue())
                        results[security_name] = values
            if event.eventType() == blpapi.Event.RESPONSE:
                break

        if len(results) == 1:
            results = results.values()[0]
        return results

def main():
    bbg = Bloomberg()
    refs = bbg.get_reference_datum('G H5 Comdty', 'FUT_CTD_CUSIP')
    print refs


if __name__ == "__main__":
    import sys

    sys.exit(main())
