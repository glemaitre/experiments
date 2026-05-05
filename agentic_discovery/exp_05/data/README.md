## Electricity load data (France)

The data is manually retrieved from the ENTSOE Open Data portal. The CSV file needs to be downloaded from an browser by clicking for the right fields in a form:

https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show?name=&defaultValue=false&viewType=TABLE&areaType=BZN&atch=false&dateTime.dateTime=12.10.2023+00:00|UTC|DAY&biddingZone.values=CTY|10YFR-RTE------C!BZN|10YFR-RTE------C&dateTime.timezone=UTC&dateTime.timezone_input=UTC#

- click the "Export Data" menu, then login / create and account;
- once logged-in, navigate back to the link above;
- make sure to select UTC time;
- tick the country of interest ("BZN|FR" in our case) in the left-hand panel;
- put a date within the year of interest (between 2020 and now);
- click the down arrow of the "Export Data" menu, and select "Total Load - Day
  Ahead / Actual (Year, CSV)".

## Weather data:

From https://open-meteo.com/en/docs/historical-forecast-api. The data was
fetched with the help of the `fetch_weather_data.py` script.

## Additional information

We have the following information at hand

- Historical weather data for 10 medium to large urban areas in France;
- Holidays and standard calendar features for France;
- Historical electricity load data for the whole of France.

All these data sources cover a time range from March 23, 2021 to May 31, 2025.

Since our maximum forecasting horizon is 24 hours, we consider that the future weather data is known at a chosen prediction time. Similarly, the holidays and calendar features are known at prediction time for any point in the future.

Therefore, exogenous features derived from the weather and calendar data can be used to engineer “future covariates”. Since the load data is our prediction target, we will can also use it to engineer “past covariates” such as lagged features and rolling aggregations. The future values of the load data (with respect to the prediction time) are used as targets for the forecasting model.