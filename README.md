# Optimization-of-AirBnB-Pricing-Startegy
The objective of this project is understanding what factors affect AirBnB rental prices the most in order to help new hosts offer both relevant and attractive deals.

## Introduction

To ensure the work on analyzing the data is going smoothly I asked myself a few questions first:

- Which areas are the cheapest, and which are the most expensive?
- Is there a difference in rental prices between properties offering instant booking and those that don’t?
- How does the cancellation policy correlate with the average rental price for similar properties?
- What factors do reviews correlate with and do they affect the final prices?
- Do properties with higher average review scores and greater number of reviews command higher rental prices?

But before I answer the questions above I’m going to get familiar with the contents of the table and adjust the data in a way facilitating the points I want to look at.


## Data configuration

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

aob = pd.read_csv('D:\\PRACTIMA\\Airbnb_Open_Data.csv',
                  low_memory=False)
```

I started with checking the size of the table.

```python
print(aob.shape)
```
> (102599, 26)

Now we can see that there’s quite a lot of data- 26 columns and 102 599 rows. 
Next, I checked the columns.

```python
print(aob.columns.values)
```
> ['id' 'NAME' 'host id' 'host_identity_verified' 'host name' 'neighbourhood group' 'neighbourhood' 'lat' 'long' 'country' 'country code' 'instant_bookable' 'cancellation_policy' 'room type' 'Construction year' 'price' 'service fee' 'minimum nights' 'number of reviews' 'last review' 'reviews per month' 'review rate number' 'calculated host listings count' 'availability 365' 'house_rules' 'license']

This shows us the table contains different types of data: dates, text, numbers, even latitudes and longitudes. In order to prepare them for further analysis I made sure what are the exact types of data in the table. 

```python
print(aob.dtypes)
```
| NAZWA KOLUMNY                    | TYP DANYCH  |
| -------------------------------- | ------------|
| NAME                             | object      |
| host id                          | int64       |
| host_identity_verified           | object      |
| host name                        | object      |
| neighbourhood group              | object      |
| neighbourhood                    | object      |
| lat                              | float64     |
| long                             | float64     |
| country                          | object      |
| country code                     | object      |
| instant_bookable                 | object      |
| cancellation_policy              | object      |
| room type                        | object      |
| Construction year                | float64     |
| price                            | object      |
| service fee                      | object      |
| minimum nights                   | float64     |
| number of reviews                | float64     |
| last review                      | object      |
| reviews per month                | float64     |
| review rate number               | float64     |
| calculated host listings count   | float64     |
| availability 365                 | float64     |
| house_rules                      | object      |
| license                          | object      |

Based on this information we can see that the prices and service fees are listed as an ‘object’ - it’s going to be easier if they were a ‘float’. Before I take care of the data conversion, I’d like to get a better knowledge of the contents of the table by printing out the first 5 rows of it.

```python
df_head = aob.head()
cols = aob.columns
first_part = df_head[cols[:13]]  
second_part = df_head[cols[13:]]

print(first_part)
print(second_part)
```
## 5 pierwszych rzędów (podzielone na dwie części)
| id       | NAME                                               | host id      | host_identity_verified | host name | neighbourhood group | neighbourhood | lat      | long      | country       | country code | instant_bookable | cancellation_policy |
|----------|----------------------------------------------------|--------------|------------------------|-----------|----------------------|---------------|----------|-----------|----------------|---------------|------------------|----------------------|
| 1001254  | Clean & quiet apt home by the park                 | 80014485718  | unconfirmed            | Madaline  | Brooklyn             | Kensington    | 40.64749 | -73.97237 | United States  | US            | False            | strict               |
| 1002102  | Skylit Midtown Castle                              | 52335172823  | verified               | Jenna     | Manhattan            | Midtown       | 40.75362 | -73.98377 | United States  | US            | False            | moderate             |
| 10024708 | THE VILLAGE OF HARLEM...NEAR ALL!                 | 78829239556  | nan                    | Elise     | Manhattan            | Harlem        | 40.80902 | -73.94190 | United States  | US            | True             | flexible             |
| 1002755  | nan                                                | 85082326012  | unconfirmed            | Garry     | Brooklyn             | Clinton Hill  | 40.68514 | -73.95976 | United States  | US            | True             | moderate             |
| 1006436  | Spacious Studio/Loft by central park               | 92073596077  | verified               | Lyndon    | Manhattan            | East Harlem   | 40.79851 | -73.94399 | United States  | US            | False            | moderate             |

| room type       | Construction year | price | service fee | minimum nights | number of reviews | last review | reviews per month | review rate number | calculated host listings count | availability 365 | house_rules                                                                                      | license |
|-----------------|-------------------|-------|-------------|----------------|--------------------|--------------|--------------------|---------------------|----------------------------------|------------------|--------------------------------------------------------------------------------------------------|---------|
| Private room    | 2020.0            | $966  | $193        | 10.0           | 9                  | 10/19/2021   | 0.21               | 4.0                 | 6.0                              | 286.0            | Clean up and treat the home the way you'd like your home treated                                  | nan     |
| Entire home/apt | 2007.0            | $917  | $84         | 15.0           | 21                 | 12/14/2022   | 0.23               | 4.0                 | 2.0                              | 278.0            | Guests (for an extra fee) are welcome, but this also needs to be confirmed                       | nan     |
| Private room    | 2005.0            | $620  | $24         | 10.0           | 9                  | nan          | nan                | 5.0                 | 10.0                             | 362.0            | I encourage you to use my kitchen, cooking and laundry facilities. No smoking, inside or outside. | nan     |
| Entire home/apt | 2005.0            | $368  | $74         | 30.0           | 270                | 7/5/2019     | 4.64               | 4.0                 | 1.0                              | 322.0            | nan                                                                                              | nan     |
| Entire home/apt | 2009.0            | $204  | $41         | 10.0           | 9                  | 11/19/2018   | 0.1                | 4.0                 | 2.0                              | 280.0            | Please no smoking in the house, porch or on the property. You can go to the nearby corner.       | nan     |

I divided the table in two showing the first 13 columns at the top, 13 at the bottom to make it easier to read. It’s a small example, but it’s enough to see that the columns containing any types of prices also have a currency sign at the beginning of it, the column ‘instant_bookable’ contains only true or false values. This is enough information for me to begin with the initial data cleaning.  

## Data cleaning

Since there’s a lot of data I decided to check if there are any duplicates based on the ‘id’ column that should be different for each one of the listed properties. Because of the size of the table I showed only the first 5 examples.

```python
aob_duplicates = aob["id"]
duplicates = aob[aob_duplicates.isin(aob_duplicates[aob_duplicates.duplicated()])]
duplicates_filtered = duplicates[['id', 'NAME', 'lat', 'long', 'price']].sort_values("id")
print(duplicates_filtered)
```

|          | id       | NAME                                                         | lat      | long      | price  |
|----------|----------|--------------------------------------------------------------|----------|-----------|--------|
| 9098     | 6026161  | Upper East Side 2 bedroom- close to Hospitals-               | 40.76222 | -73.96030 | $105   |
| 102474   | 6026161  | Upper East Side 2 bedroom- close to Hospitals-               | 40.76222 | -73.96030 | $105   |
| 102475   | 6026714  | Close to East Side Hospitals- Modern 2 Bedroom...            | 40.76249 | -73.96217 | $285   |
| 9099     | 6026714  | Close to East Side Hospitals- Modern 2 Bedroom...            | 40.76249 | -73.96217 | $285   |
| 9100     | 6027266  | ACADIA Spacious 2 Bedroom Apt - Close to Hospitals...        | 40.76021 | -73.96157 | $586   |
| 102239   | 35606797 | Bright and Beautiful Top Floor Two Bedrooms                  | 40.68383 | -73.99281 | $1,027 |
| 62658    | 35607349 | Modern & Bright Queen Bedroom Midtown East                   | 40.76132 | -73.96064 | $141   |
| 102240   | 35607349 | Modern & Bright Queen Bedroom Midtown East                   | 40.76132 | -73.96064 | $141   |
| 102241   | 35607902 | Modern NEW Room\|PRIVATE BATHROOM                            | 40.68990 | -73.94074 | $284   |
| 62659    | 35607902 | Modern NEW Room\|PRIVATE BATHROOM                            | 40.68990 | -73.94074 | $284   |

> [1082 rows x 5 columns]

There’s 1082 duplicated rows so in order to make it more readable and make sure the analysis is accurate I deleted them and checked if the size of the table changed.

```python
aob.drop_duplicates(subset=['id'], inplace=True)
print(aob.shape)
```
> (101760, 26)

Next I’ve checked if there are any missing values in each column.

```python
aob.isnull().sum()
```
| NAZWA KOLUMNY                    | TYP DANYCH  |
| -------------------------------- | ------------|
| id                               | 0         |
| NAME                             | 250         |
| host id                          | 0           |
| host_identity_verified           | 289         |
| host name                        | 406         |
| neighbourhood group              | 29          |
| neighbourhood                    | 16          |
| lat                              | 8           |
| long                             | 8           |
| country                          | 532         |
| country code                     | 131         |
| instant_bookable                 | 105         |
| cancellation_policy              | 76          |
| room type                        | 0           |
| Construction year                | 214         |
| price                            | 247         |
| service fee                      | 273         |
| minimum nights                   | 409         |
| number of reviews                | 183         |
| last review                      | 15893       |
| reviews per month                | 15879       |
| review rate number               | 326         |
| calculated host listings count   | 319         |
| availability 365                 | 448         |
| house_rules                      | 52131       |
| license                          | 102597      |
| dtype: int64                     |             |

As we can see above, there are a few missing values, especially in the ‘price’ column. Since this project is all about the rental prices I decided to get rid of the columns that don’t have this information listed.

```python
aob.dropna(subset=['price'], inplace=True)
```

We can also notice that there are missing values in columns ‘country’ and ‘country code’ but since all the data is from the United States I just filled them in.

```python
aob.fillna ({
    'country': 'United States',
    'country code': 'US'}, inplace=True)
```

Moving on, I’ve checked the exact values in the ‘availability 365’ column to make sure everything is correct there.

```python
unique_365 = aob['availability 365'].unique()
sorted_365 = np.sort(unique_365)
print(sorted_365.tolist())
```
> [-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0, 170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0, 180.0, 181.0, 182.0, 183.0, 184.0, 185.0, 186.0, 187.0, 188.0, 189.0, 190.0, 191.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0, 198.0, 199.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0, 218.0, 219.0, 220.0, 221.0, 222.0, 223.0, 224.0, 225.0, 226.0, 227.0, 228.0, 229.0, 230.0, 231.0, 232.0, 233.0, 234.0, 235.0, 236.0, 237.0, 238.0, 239.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 256.0, 257.0, 258.0, 259.0, 260.0, 261.0, 262.0, 263.0, 264.0, 265.0, 266.0, 267.0, 268.0, 269.0, 270.0, 271.0, 272.0, 273.0, 274.0, 275.0, 276.0, 277.0, 278.0, 279.0, 280.0, 281.0, 282.0, 283.0, 284.0, 285.0, 286.0, 287.0, 288.0, 289.0, 290.0, 291.0, 292.0, 293.0, 294.0, 295.0, 296.0, 297.0, 298.0, 299.0, 300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0, 308.0, 309.0, 310.0, 311.0, 312.0, 313.0, 314.0, 315.0, 316.0, 317.0, 318.0, 319.0, 320.0, 321.0, 322.0, 323.0, 324.0, 325.0, 326.0, 327.0, 328.0, 329.0, 330.0, 331.0, 332.0, 333.0, 334.0, 335.0, 336.0, 337.0, 338.0, 339.0, 340.0, 341.0, 342.0, 343.0, 344.0, 345.0, 346.0, 347.0, 348.0, 349.0, 350.0, 351.0, 352.0, 353.0, 354.0, 355.0, 356.0, 357.0, 358.0, 359.0, 360.0, 361.0, 362.0, 363.0, 364.0, 365.0, 366.0, 367.0, 368.0, 369.0, 370.0, 371.0, 372.0, 373.0, 374.0, 375.0, 376.0, 377.0, 378.0, 379.0, 380.0, 381.0, 382.0, 383.0, 384.0, 385.0, 386.0, 387.0, 388.0, 389.0, 390.0, 391.0, 392.0, 393.0, 394.0, 395.0, 396.0, 397.0, 398.0, 399.0, 400.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0, 408.0, 409.0, 410.0, 411.0, 412.0, 413.0, 414.0, 415.0, 416.0, 417.0, 418.0, 419.0, 420.0, 421.0, 422.0, 423.0, 424.0, 425.0, 426.0, 3677.0, nan]

It’s clear that there are values below 0 and above 365. Since it’s impossible to countercheck the proper information I filled them in as ‘NaN’ so that they don’t disturb the results.

```python
aob.loc[(aob['availability 365'] < 0) | (aob['availability 365'] > 365), 'availability 365'] = np.nan

cleared_365 = aob['availability 365'].unique()
cleared_365_s = np.sort(cleared_365)
print(cleared_365_s.tolist())
```
> [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0, 170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0, 180.0, 181.0, 182.0, 183.0, 184.0, 185.0, 186.0, 187.0, 188.0, 189.0, 190.0, 191.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0, 198.0, 199.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0, 218.0, 219.0, 220.0, 221.0, 222.0, 223.0, 224.0, 225.0, 226.0, 227.0, 228.0, 229.0, 230.0, 231.0, 232.0, 233.0, 234.0, 235.0, 236.0, 237.0, 238.0, 239.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 256.0, 257.0, 258.0, 259.0, 260.0, 261.0, 262.0, 263.0, 264.0, 265.0, 266.0, 267.0, 268.0, 269.0, 270.0, 271.0, 272.0, 273.0, 274.0, 275.0, 276.0, 277.0, 278.0, 279.0, 280.0, 281.0, 282.0, 283.0, 284.0, 285.0, 286.0, 287.0, 288.0, 289.0, 290.0, 291.0, 292.0, 293.0, 294.0, 295.0, 296.0, 297.0, 298.0, 299.0, 300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0, 308.0, 309.0, 310.0, 311.0, 312.0, 313.0, 314.0, 315.0, 316.0, 317.0, 318.0, 319.0, 320.0, 321.0, 322.0, 323.0, 324.0, 325.0, 326.0, 327.0, 328.0, 329.0, 330.0, 331.0, 332.0, 333.0, 334.0, 335.0, 336.0, 337.0, 338.0, 339.0, 340.0, 341.0, 342.0, 343.0, 344.0, 345.0, 346.0, 347.0, 348.0, 349.0, 350.0, 351.0, 352.0, 353.0, 354.0, 355.0, 356.0, 357.0, 358.0, 359.0, 360.0, 361.0, 362.0, 363.0, 364.0, 365.0, nan]

While getting to know the table better I also checked the unique values in ‘neighbourhood group’, ‘neighbourhood’ and ‘room type’.

```python
print(f"neighbourhood groups: {aob['neighbourhood group'].unique()}")
print(f"room types: {aob['room type'].unique()}")
print(f"neighbourhoods: {aob['neighbourhood'].unique()}")
```
> neighbourhood groups: ['Brooklyn' 'Manhattan' 'brookln' 'manhatan' 'Queens' nan 'Staten Island'
 'Bronx']

> room types: ['Private room' 'Entire home/apt' 'Shared room' 'Hotel room']

> neighbourhoods: ['Kensington' 'Midtown' 'Harlem' 'Clinton Hill' 'East Harlem' 'Murray Hill' 'Bedford-Stuyvesant' "Hell's Kitchen" 'Upper West Side' 'Chinatown' 'South Slope' 'West Village' 'Williamsburg' 'Fort Greene' 'Chelsea' 'Crown Heights' 'Park Slope' 'Windsor Terrace' 'Inwood' 'East Village' 'Greenpoint' 'Bushwick' 'Flatbush' 'Lower East Side' 'Prospect-Lefferts Gardens' 'Long Island City' 'Kips Bay' 'SoHo' 'Upper East Side' 'Prospect Heights' 'Washington Heights' 'Woodside' 'Brooklyn Heights' 'Carroll Gardens' 'Gowanus' 'Flatlands' 'Cobble Hill' 'Flushing' 'Boerum Hill' 'Sunnyside' 'DUMBO' 'St. George' 'Highbridge' 'Financial District' 'Ridgewood' 'Morningside Heights' 'Jamaica' 'Middle Village' 'NoHo' 'Ditmars Steinway' 'Flatiron District' 'Roosevelt Island' 'Greenwich Village' 'Little Italy' 'East Flatbush' 'Tompkinsville' 'Astoria' 'Clason Point' 'Eastchester' 'Kingsbridge' 'Two Bridges' 'Queens Village' 'Rockaway Beach' 'Forest Hills' 'Nolita' 'Woodlawn' 'University Heights' 'Gravesend' 'Gramercy' 'Allerton' nan 'East New York' 'Theater District' 'Concourse Village' 'Sheepshead Bay' 'Emerson Hill' 'Fort Hamilton' 'Bensonhurst' 'Tribeca' 'Shore Acres' 'Sunset Park' 'Concourse' 'Elmhurst' 'Brighton Beach' 'Jackson Heights' 'Cypress Hills' 'St. Albans' 'Arrochar' 'Rego Park' 'Wakefield' 'Clifton' 'Bay Ridge' 'Graniteville' 'Spuyten Duyvil' 'Stapleton' 'Briarwood' 'Ozone Park' 'Columbia St' 'Vinegar Hill' 'Mott Haven' 'Longwood' 'Canarsie' 'Battery Park City' 'Civic Center' 'East Elmhurst' 'New Springville' 'Morris Heights' 'Arverne' 'Cambria Heights' 'Tottenville' 'Mariners Harbor' 'Concord' 'Borough Park' 'Bayside' 'Downtown Brooklyn' 'Port Morris' 'Fieldston' 'Kew Gardens' 'Midwood' 'College Point' 'Mount Eden' 'City Island' 'Glendale' 'Port Richmond' 'Red Hook' 'Richmond Hill' 'Bellerose' 'Maspeth' 'Williamsbridge' 'Soundview' 'Woodhaven' 'Woodrow' 'Co-op City' 'Stuyvesant Town' 'Parkchester' 'North Riverdale' 'Dyker Heights' 'Bronxdale' 'Sea Gate' 'Riverdale' 'Kew Gardens Hills' 'Bay Terrace' 'Norwood' 'Claremont Village' 'Whitestone' 'Fordham' 'Bayswater' 'Navy Yard' 'Brownsville' 'Eltingville' 'Fresh Meadows' 'Mount Hope' 'Lighthouse Hill' 'Springfield Gardens' 'Howard Beach' 'Belle Harbor' 'Jamaica Estates' 'Van Nest' 'Morris Park' 'West Brighton' 'Far Rockaway' 'South Ozone Park' 'Tremont' 'Corona' 'Great Kills' 'Manhattan Beach' 'Marble Hill' 'Dongan Hills' 'Castleton Corners' 'East Morrisania' 'Hunts Point' 'Neponsit' 'Pelham Bay' 'Randall Manor' 'Throgs Neck' 'Todt Hill' 'West Farms' 'Silver Lake' 'Morrisania' 'Laurelton' 'Grymes Hill' 'Holliswood' 'Pelham Gardens' 'Belmont' 'Rosedale' 'Edgemere' 'New Brighton' 'Midland Beach' 'Baychester' 'Melrose' 'Bergen Beach' 'Richmondtown' 'Howland Hook' 'Schuylerville' 'Coney Island' 'New Dorp Beach' "Prince's Bay" 'South Beach' 'Bath Beach' 'Jamaica Hills' 'Oakwood' 'Castle Hill' 'Hollis' 'Douglaston' 'Huguenot' 'Olinville' 'Edenwald' 'Grant City' 'Westerleigh' 'Bay Terrace, Staten Island' 'Westchester Square' 'Little Neck' 'Fort Wadsworth' 'Rosebank' 'Unionport' 'Mill Basin' 'Arden Heights' "Bull's Head" 'New Dorp' 'Rossville' 'Breezy Point' 'Willowbrook' 'Glen Oaks' 'Gerritsen Beach' 'Chelsea, Staten Island']

The ‘neighbourhood’ column is much more detailed than the ‘neighbourhood group’ one and there are four types of properties. We can also see that the values in ‘neighbourhood’ column contain spelling mistakes so I fixed them.

```python
aob['neighbourhood group'].replace(['brookln'], 'Brooklyn', inplace=True)
aob['neighbourhood group'].replace(['manhatan'], 'Manhattan', inplace=True)

print(f"neighbourhood groups: {aob['neighbourhood group'].unique()}")
```
> neighbourhood groups: ['Brooklyn' 'Manhattan' 'Queens' 'Bronx' 'Staten Island']

Now that I’ve checked if everything is correct in the chosen columns I could finally take care of the data conversion and make the contents of the table easier to work with.

```python
convert_to_string = ['id', 'NAME', 'host id', 'host_identity_verified', 'host name',
                      'neighbourhood group', 'neighbourhood', 'country',
                      'country code', 'cancellation_policy', 'room type',
                      'price', 'service fee',
                      'Construction year', 'last review', 'house_rules']

aob[convert_to_string] = aob[convert_to_string].astype('str')

if 'instant_bookable' in aob.columns:
    aob['instant_bookable'] = aob['instant_bookable'].astype('bool')

aob['price'] = aob['price'].replace({'\\$': ''}, regex=True)
aob['service fee'] = aob['service fee'].replace({'\\$': ''}, regex=True)
aob['price'] = pd.to_numeric(aob['price'], errors='coerce')
aob['service fee'] = pd.to_numeric(aob['service fee'], errors='coerce')
```

Then I’ve made sure all of the data is correct.

```python
print(aob.dtypes)
```
| NAZWA KOLUMNY                    | TYP DANYCH  |
| -------------------------------- | ------------|
| id                               | object      |
| NAME                             | object      |
| host id                          | object      |
| host_identity_verified           | object      |
| host name                        | object      |
| neighbourhood group              | object      |
| neighbourhood                    | object      |
| lat                              | float64     |
| long                             | float64     |
| country                          | object      |
| country code                     | object      |
| instant_bookable                 | bool        |
| cancellation_policy              | object      |
| room type                        | object      |
| Construction year                | float64     |
| price                            | float64     |
| service fee                      | float64     |
| minimum nights                   | float64     |
| number of reviews                | float64     |
| last review                      | object      |
| reviews per month                | float64     |
| review rate number               | float64     |
| calculated host listings count   | float64     |
| availability 365                 | float64     |
| house_rules                      | object      |
| license                          | object      |
| dtype: object                    |             |

Now I can continue getting familiar with the data in the table in order to answer the questions asked at the beginning of my analysis.

## Exploratory data analysis

I counted how many properties are in each neighbourhood group, as well as how many properties of each type.

```python
aob['neighbourhood group'].value_counts()
```
| neighbourhood group              |             |
| -------------------------------- | ------------|
| Manhattan                        | 30662       |
| Brooklyn                         | 28566       |
| Queens                           | 10067       |
| Bronx                            | 2182        |
| Staten Island                    | 821         |
| Name: count, dtype: int64        | object      |

```python
aob['room type'].value_counts()
```
| room type                        |             |
| -------------------------------- | ------------|
| Entire home/apt                  | 39185       |
| Private room                     | 31537       |
| Shared room                      | 1477        |
| Hotel room                       | 90          |
| Name: count, dtype: int64        |             |

Based on this result we can see that the most properties are located in Manhattan and Brooklyn, the least are located in Staten Island. As for the types of properties, the most accessible are entire houses or apartments, and the least are hotel rooms. 
Next I've checked the amount of each room types in each neighbourhood group.

```python
room_type_pivot = aob.pivot_table(
    index='neighbourhood group',
    columns='room type',
    aggfunc='size',
    fill_value=0
)
print(room_type_pivot)
```
| Neighbourhood Group  | Entire home/apt | Hotel room | Private room | Shared room |
|----------------------|-----------------|------------|--------------|-------------|
| **Bronx**            | 1022            | 0          | 1573         | 117         |
| **Brooklyn**         | 20575           | 8          | 20435        | 825         |
| **Manhattan**        | 26473           | 100        | 16313        | 907         |
| **Queens**           | 5146            | 8          | 7751         | 362         |
| **Staten Island**    | 474             | 0          | 466          | 15          |
| **nan**              | 10              | 0          | 18           | 0           |

This allows us to conclude that:

- in Bronx the most available property type to find is private room, the least is shared room and there are no hotel rooms;
- in Brooklyn the most available property types are entire home/apt and private room, the least is hotel room;
- in Manhattan the most available property type is entire home/apt, the least is hotel room;
- in Queens the most available property type is private room, the least is hotel room;
- in Staten Island the most available property types are entire home/apt and private room, the least shared room, and there are no hotel rooms.

While analyzing the types of properties I also looked at the price ranges of each one of them.

```python
rooms = aob.groupby(['room type'])['price'].agg(['min', 'max', 'mean']).reset_index()
rooms.columns = ['Room type', 'Minimum Price', 'Maximum Price', 'Mean Price']
print(rooms)
```
| Room Type         | Min Price | Max Price | Average Price |
|-------------------|-----------|-----------|---------------|
| **Entire home/apt** | 50.0      | 999.0     | 526.88        |
| **Hotel room**      | 50.0      | 994.0     | 574.69        |
| **Private room**    | 50.0      | 999.0     | 523.43        |
| **Shared room**     | 50.0      | 998.0     | 511.57        |

The highest average price is presented by hotel rooms, the lowest shared rooms but the differences in prices between each room type are very small, price ranges are also similar. I decided to take neighbourhood groups and neighbourhoods into account to get a better perspective.

```python
price_summary = aob.groupby(['room type', 'neighbourhood group'])['price'].agg(['min', 'max', 'mean']).reset_index()
price_summary.columns = ['Room Type', 'Neighbourhood Group', 'Minimum Price', 'Maximum Price', 'Mean Price']
print(price_summary)
```
| Room Type           | Neighbourhood Group  | Minimum Price | Maximum Price | Mean Price   |
|---------------------|----------------------|---------------|---------------|--------------|
| **Entire home/apt** | Bronx                | 50.0          | 996.0         | 531.56       |
| **Entire home/apt** | Brooklyn             | 50.0          | 999.0         | 531.02       |
| **Entire home/apt** | Manhattan            | 50.0          | 999.0         | 525.01       |
| **Entire home/apt** | Queens               | 50.0          | 999.0         | 517.30       |
| **Entire home/apt** | Staten Island        | 50.0          | 998.0         | 548.42       |
| **Hotel room**      | Brooklyn             | 246.0         | 907.0         | 664.33       |
| **Hotel room**      | Manhattan            | 50.0          | 994.0         | 578.87       |
| **Hotel room**      | Queens               | 242.0         | 912.0         | 486.00       |
| **Private room**    | Bronx                | 50.0          | 998.0         | 532.80       |
| **Private room**    | Brooklyn             | 50.0          | 999.0         | 524.20       |
| **Private room**    | Manhattan            | 50.0          | 999.0         | 520.49       |
| **Private room**    | Queens               | 50.0          | 999.0         | 526.16       |
| **Private room**    | Staten Island        | 50.0          | 996.0         | 510.23       |
| **Shared room**     | Bronx                | 63.0          | 978.0         | 507.88       |
| **Shared room**     | Brooklyn             | 51.0          | 998.0         | 494.42       |
| **Shared room**     | Manhattan            | 50.0          | 998.0         | 520.18       |
| **Shared room**     | Queens               | 50.0          | 998.0         | 521.69       |
| **Shared room**     | Staten Island        | 182.0         | 967.0         | 619.82       |

This result isn't easy to read, so I created a chart.

```python
plt.figure(figsize=(12, 6))

sns.barplot(data=price_summary, x='Neighbourhood Group', y='Mean Price', hue='Room Type')

plt.title('Mean Price by Room Type and Neighbourhood Group')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Mean Price')
plt.xticks(rotation=45) 
plt.legend(title='Room Type')

plt.tight_layout()
plt.show()
```
![wykres1](https://github.com/user-attachments/assets/65f9801f-7c11-421a-a83f-9b4fafd212b5)

Now we can see straight away that the prices are quite even. The only exceptions are hotel rooms in Brooklyn and shared rooms in Staten Island. 

I also checked the average prices for each neighbourhood.

```python
average_prices = aob.groupby(['neighbourhood group', 'neighbourhood'])['price'].mean().reset_index()

sorted_average_prices = average_prices.sort_values(by=['neighbourhood group', 'price'])

pd.set_option('display.max_rows', None)

print(sorted_average_prices)
```
| neighbourhood group | neighbourhood           | price      |
|---------------------|-------------------------|------------|
| Bronx               | Spuyten Duyvil          | 293.444444 |
| Bronx               | Co-op City              | 360.750000 |
| Bronx               | Morrisania              | 385.200000 |
| Bronx               | Baychester              | 395.428571 |
| Bronx               | Unionport               | 429.538462 |
| Bronx               | Van Nest                | 443.166667 |
| Bronx               | Tremont                 | 455.523810 |
| Bronx               | Allerton                | 461.305085 |
| Bronx               | West Farms              | 463.166667 |
| Bronx               | Bronxdale               | 465.740741 |
| Bronx               | Concourse Village      | 475.153846 |
| Bronx               | Castle Hill             | 479.133333 |
| Bronx               | Schuylerville           | 479.806452 |
| Bronx               | Woodlawn                | 489.869565 |
| Bronx               | Pelham Gardens          | 500.241379 |
| Bronx               | Williamsbridge          | 504.327586 |
| Bronx               | Kingsbridge             | 505.684211 |
| Bronx               | Fieldston               | 506.933333 |
| Bronx               | Mott Haven              | 509.169811 |
| Bronx               | Norwood                 | 509.547619 |
| Bronx               | Edenwald                | 514.925926 |
| Bronx               | Port Morris             | 521.493333 |
| Bronx               | Mount Hope              | 527.805556 |
| Bronx               | Claremont Village      | 534.577778 |
| Bronx               | Melrose                 | 539.642857 |
| Bronx               | Westchester Square     | 541.055556 |
| Bronx               | East Morrisania        | 542.666667 |
| Bronx               | Concourse              | 544.012500 |
| Bronx               | Wakefield              | 547.596154 |
| Bronx               | Clason Point           | 550.142857 |
| Bronx               | Hunts Point            | 555.380952 |
| Bronx               | Fordham                 | 555.921739 |
| Bronx               | Eastchester             | 557.600000 |
| Bronx               | City Island            | 558.766667 |
| Bronx               | Parkchester            | 567.000000 |
| Bronx               | Highbridge             | 570.500000 |
| Bronx               | University Heights     | 575.129032 |
| Bronx               | North Riverdale        | 586.000000 |
| Bronx               | Morris Heights         | 595.750000 |
| Bronx               | Longwood                | 600.305556 |
| Bronx               | Throgs Neck            | 600.828571 |
| Bronx               | Soundview               | 601.000000 |
| Bronx               | Mount Eden             | 603.500000 |
| Bronx               | Riverdale              | 604.181818 |
| Bronx               | Pelham Bay             | 612.000000 |
| Bronx               | Belmont                | 618.851852 |
| Bronx               | Morris Park            | 637.966667 |
| Bronx               | Olinville              | 669.125000 |
| Brooklyn            | Navy Yard              | 423.153846 |
| Brooklyn            | Dyker Heights          | 454.866667 |
| Brooklyn            | Brooklyn Heights       | 459.296053 |
| Brooklyn            | DUMBO                   | 459.459459 |
| Brooklyn            | Sea Gate               | 459.857143 |
| Brooklyn            | Red Hook               | 472.592233 |
| Brooklyn            | Fort Hamilton          | 473.376471 |
| Brooklyn            | Bergen Beach           | 475.291667 |
| Brooklyn            | Cobble Hill            | 478.990476 |
| Brooklyn            | Windsor Terrace        | 487.692683 |
| Brooklyn            | Borough Park           | 494.350993 |
| Brooklyn            | Kensington             | 497.145729 |
| Brooklyn            | Bensonhurst            | 499.481818 |
| Brooklyn            | Gravesend              | 501.384615 |
| Brooklyn            | Vinegar Hill           | 504.794872 |
| Brooklyn            | Fort Greene            | 505.472381 |
| Brooklyn            | Bay Ridge              | 513.695000 |
| Brooklyn            | East New York          | 514.520362 |
| Brooklyn            | Park Slope             | 515.220430 |
| Brooklyn            | Brighton Beach         | 517.149123 |
| Brooklyn            | Sheepshead Bay         | 517.196507 |
| Brooklyn            | Greenpoint             | 521.787659 |
| Brooklyn            | Crown Heights          | 522.789530 |
| Brooklyn            | Bushwick               | 526.453890 |
| Brooklyn            | Canarsie               | 527.091241 |
| Brooklyn            | Prospect-Lefferts Gardens | 527.239394 |
| Brooklyn            | Flatlands              | 527.424460 |
| Brooklyn            | Bedford-Stuyvesant     | 528.562128 |
| Brooklyn            | Midwood                | 528.576642 |
| Brooklyn            | Clinton Hill           | 531.172131 |
| Brooklyn            | Williamsburg           | 531.260923 |
| Brooklyn            | Gowanus                | 536.074733 |
| Brooklyn            | Sunset Park            | 537.763705 |
| Brooklyn            | Flatbush               | 541.963450 |
| Brooklyn            | Prospect Heights       | 542.750678 |
| Brooklyn            | Cypress Hills          | 544.633484 |
| Brooklyn            | East Flatbush          | 544.750000 |
| Brooklyn            | South Slope            | 545.967164 |
| Brooklyn            | Boerum Hill            | 546.206731 |
| Brooklyn            | Downtown Brooklyn      | 547.580645 |
| Brooklyn            | Carroll Gardens        | 554.808989 |
| Brooklyn            | Brownsville            | 569.504854 |
| Brooklyn            | Manhattan Beach        | 571.571429 |
| Brooklyn            | Bath Beach             | 595.000000 |
| Brooklyn            | Mill Basin             | 598.000000 |
| Brooklyn            | Coney Island           | 623.181818 |
| Brooklyn            | Columbia St            | 626.566038 |
| Brooklyn            | Gerritsen Beach        | 694.333333 |
| Manhattan           | Marble Hill            | 454.866667 |
| Manhattan           | Murray Hill            | 492.158287 |
| Manhattan           | SoHo                    | 496.967172 |
| Manhattan           | Tribeca                 | 501.595000 |
| Manhattan           | Little Italy           | 503.537572 |
| Manhattan           | Gramercy                | 507.807198 |
| Manhattan           | East Village           | 511.989270 |
| Manhattan           | Flatiron District      | 512.357895 |
| Manhattan           | East Harlem            | 512.886850 |
| Manhattan           | Nolita                  | 517.588028 |
| Manhattan           | Roosevelt Island       | 518.520000 |
| Manhattan           | Upper East Side        | 520.123890 |
| Manhattan           | Financial District     | 520.288981 |
| Manhattan           | Battery Park City      | 520.358209 |
| Manhattan           | Harlem                 | 521.870695 |
| Manhattan           | Morningside Heights   | 522.662651 |
| Manhattan           | Washington Heights     | 524.396518 |
| Manhattan           | Midtown                | 525.187984 |
| Manhattan           | Hell's Kitchen         | 525.633763 |
| Manhattan           | Upper West Side        | 526.613411 |
| Manhattan           | Civic Center           | 527.418182 |
| Manhattan           | West Village           | 529.341250 |
| Manhattan           | Theater District       | 529.650919 |
| Manhattan           | Kips Bay               | 531.175047 |
| Manhattan           | Chelsea                | 531.376304 |
| Manhattan           | Lower East Side        | 533.637205 |
| Manhattan           | Two Bridges            | 542.722222 |
| Manhattan           | Chinatown              | 548.300000 |
| Manhattan           | Inwood                 | 552.046980 |
| Manhattan           | Greenwich Village      | 558.534626 |
| Manhattan           | Stuyvesant Town        | 586.114286 |
| Manhattan           | NoHo                    | 616.863636 |
| Queens              | Breezy Point           | 309.888889 |
| Queens              | Bayside                | 449.744681 |
| Queens              | Belle Harbor           | 456.521739 |
| Queens              | Maspeth                | 460.565714 |
| Queens              | Douglaston             | 462.238095 |
| Queens              | Howard Beach           | 472.642857 |
| Queens              | Fresh Meadows          | 476.132353 |
| Queens              | Woodhaven              | 476.387597 |
| Queens              | Richmond Hill          | 483.133758 |
| Queens              | Forest Hills           | 485.856383 |
| Queens              | Neponsit               | 486.250000 |
| Queens              | Rosedale               | 488.780488 |
| Queens              | South Ozone Park       | 498.752941 |
| Queens              | Bay Terrace            | 500.000000 |
| Queens              | Ditmars Steinway       | 502.395683 |
| Queens              | Springfield Gardens    | 504.740741 |
| Queens              | Laurelton              | 505.200000 |
| Queens              | Bayswater              | 505.406250 |
| Queens              | Sunnyside              | 505.583519 |
| Queens              | Middle Village         | 508.716981 |
| Queens              | Rego Park              | 515.782609 |
| Queens              | Jackson Heights        | 518.476974 |
| Queens              | Hollis                 | 520.333333 |
| Queens              | Flushing               | 522.770062 |
| Queens              | Long Island City       | 522.879937 |
| Queens              | Jamaica Estates        | 523.756757 |
| Queens              | Astoria                | 525.779693 |
| Queens              | Jamaica                | 525.879518 |
| Queens              | Whitestone             | 528.000000 |
| Queens              | Far Rockaway           | 528.528302 |
| Queens              | Jamaica Hills          | 528.875000 |
| Queens              | Kew Gardens            | 530.540000 |
| Queens              | Ridgewood              | 532.745614 |
| Queens              | Ozone Park             | 535.055056 |
| Queens              | Forest Park            | 535.201053 |
| Queens              | South Richmond Hill    | 536.360000 |
| Queens              | College Point          | 536.400000 |
| Queens              | Glen Oaks              | 537.461538 |
| Queens              | Astoria Heights        | 540.716981 |
| Queens              | College Point          | 541.000000 |
| Queens              | Woodside               | 542.722222 |
| Queens              | Corona                 | 543.051725 |
| Queens              | Hillcrest              | 543.352273 |
| Queens              | South Jamaica          | 544.038462 |
| Queens              | Broad Channel          | 546.000000 |

There's a lot of results but we can clearly see that even though the differences between the neighbourhood groups are minimal, the differences between each neighbourhood are much bigger.

- The average prices in Bronx vary between $293.44 in Spuyten Duyvill and $669.12 in Olinville.
- The average prices in Brooklyn vary between $423.15 in Navy Yard and $694.33 in Gerritsen Beach.
- The average prices in Manhattan vary between $454.86 in Marble Hill and $616.86 in NoHo.
- The average prices in Queens vary between $309.88 w dzielnicyin Breezy Point and $691.66 in Little Neck.
- The average prices in Staten Island vary between $78.00 in Woodrow and $757.92 in New Dorp.

Going back to neighbourhood groups, I checked the price ranges between room types.

```python
plt.figure(figsize=(12, 6))

sns.boxplot(data=aob, x='neighbourhood group', y='price', hue='room type')

plt.title('Price Distribution by Room Type and Neighbourhood Group')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Price')
plt.xticks(rotation=45)  
plt.legend(title='Room Type')

plt.tight_layout()
plt.show()
```
![wykres2](https://github.com/user-attachments/assets/146ae577-552c-4a09-a46a-8a67370f97aa)

As we can see, the hotel rooms in Brooklyn have slightly higher prices, but overall the prices are very similar.

Holding onto the influence of localization on the price I moved to analyzing the rules of renting, starting from the availability for instant booking.

```python
plt.figure(figsize=(12, 6))
sns.barplot(
    data=instantbooking,
    x='neighbourhood group',
    y='Mean Price',
    hue='instant bookable',
    palette=['#BC4749', '#6A994E']  
)
plt.title('Average price by the possibility of instant booking')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Price ($)')
plt.legend(title='Instant Bookable')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```
![wykres3](https://github.com/user-attachments/assets/47f26e09-9a03-482a-8044-78d95d95b3ec)

This chart allows us to see that the availability for instant booking doesn't have a big impact on the prices in each of the neighbourhood groups.

Keeping that result in mind it's also worth looking at the cancellation policy and its' influence.

```python
plt.figure(figsize=(12, 6))
sns.barplot(
    data=cancellationpolicy,
    x='Neighbourhood group',
    y='Mean Price',
    hue='Cancellation policy',
    palette=['#FAA307', '#E85D04', '#9D0208', '#F4D58D']  
)
plt.title('Average price by the cancellation policy')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Price ($)')
plt.legend(title='Cancellation policy')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```
![wykres4](https://github.com/user-attachments/assets/ac81b66a-14ae-4bf8-8873-d0adfcb0c839)

The chart clearly shows that, in most cases, cancellation policy doesn't have any impact on the final renting price, except for Queens, Staten Island and Manhattan. In the first two neighbourhood groups the properties without clear cancellation policies have the highest prices, in the last one the prices are slightly lower.

Next, I've taken into account the number of days the properties are available for rental in a year.

```python
availability = aob.groupby(['availability 365'])['price'].agg(['mean']).reset_index()
availability.columns = ['availability 365', 'Mean Price']

# Create a line graph
plt.figure(figsize=(25, 6))
plt.plot(availability['availability 365'], availability['Mean Price'], color='#386641', linestyle='-')

# Add labels and title
plt.xlabel('Availability (Days)')
plt.ylabel('Mean Price')
plt.title('Mean Price by Availability (365 Days)')

# Set x-ticks to show only every 50 days
plt.xticks(range(0, 366, 50))  # Adjust the range as needed

# Add grid for better readability
plt.grid()

# Show the plot
plt.tight_layout()
plt.show()
```
![wykres5](https://github.com/user-attachments/assets/32b7c3e7-197f-4090-9e5a-1dc1a4eb351a)

At the very end I've looked at the reviews, starting from separating the properties by neighbourhood groups.

```python
colors = {
    'Bronx': '#1f77b4',
    'Brooklyn': '#ff7f0e',
    'Manhattan': '#2ca02c',
    'Queens': '#d62728',
    'Staten Island': '#9467bd'
}

plt.figure(figsize=(10, 6))

for group in reviews['Neighbourhood group'].unique():
    group_data = reviews[reviews['Neighbourhood group'] == group]
    plt.plot(group_data['Review rate'], group_data['Mean price'],
             marker='o', linestyle='-', color=colors.get(group, '#6A4C93'), label=group)

plt.xlabel('Review Rate')
plt.ylabel('Mean Price')
plt.title('Mean Price by Review Rate and Neighbourhood Group')

plt.yticks(range(0, int(reviews['Mean price'].max()) + 100, 100))
plt.ylim(400, 700)

plt.legend(title='Neighbourhood Group')

plt.grid()
 
plt.tight_layout()
plt.show()
```
![wykres6](https://github.com/user-attachments/assets/9cadfd97-65a9-4af3-9370-596a71afda15)

As we can see, the prices are once again pretty even. The bigest differences are visible in Staten Island, where the properties with the lowest review rate have the highest price, and vice versa in Bronx.

Having in mind the influence of cancellation policies I've decided to take a look at the possible connection between them and the review rates.

```python
policy_comparison = aob.groupby('cancellation_policy').agg({
    'price': 'mean',
    'review rate number': 'mean'
}).reset_index()

policy_comparison.columns = ['Cancellation Policy', 'Average Price', 'Average Review Rate Number']

print(policy_comparison)
```
| Cancellation Policy | Average Price | Average Review Rate Number |
|---------------------|---------------|-----------------------------|
| flexible            | 526.218808    | 3.296807                    |
| moderate            | 525.493277    | 3.292034                    |
| nan                 | 536.384615    | 3.272727                    |
| strict              | 523.629626    | 3.286031                    |

Just by this result we could conclude that there's no correlation between cancellation policy, average price and an average review rate, so I compared the neighbourhood groups as well. I made a heatmap presenting the average price by cancellation policy and neighbourhood group, and then the average rating by cancellation policy and the neighbourhood group.

```python
policy_comparison = aob.groupby(['cancellation_policy', 'neighbourhood group']).agg({
    'price': 'mean',
    'review rate number': 'mean'
}).reset_index()

policy_comparison.columns = ['Cancellation policy', 'Neighbourhood group', 'Average price', 'Average review rate number']

price_pivot = policy_comparison.pivot(index='Cancellation policy', columns='Neighbourhood group', values='Average price')

review_rate_pivot = policy_comparison.pivot(index='Cancellation policy', columns='Neighbourhood group', values='Average review rate number')

price_pivot = policy_comparison.pivot(index='Cancellation policy', columns='Neighbourhood group', values='Average price')

plt.figure(figsize=(10, 6))
sns.heatmap(price_pivot, annot=True, fmt=".1f", cmap='coolwarm', cbar_kws={'label': 'Average price'})
plt.title('Heatmap of average price by cancellation policy and neighbourhood group')
plt.xlabel('Neighbourhood group')
plt.ylabel('Cancellation policy')
plt.show()
```
![wykres7](https://github.com/user-attachments/assets/cb314e36-c914-480a-bead-81ec55cfa9b8)

Looking at this heatmap we can see that the properties located in Manhattan without cancellation policy stated have the lowest prices and the properties located in Queens and Staten Island without cancellation policy stated have the highest prices, which just confirms our previous conclusion.

```python
review_rate_pivot = policy_comparison.pivot(index='Cancellation policy', columns='Neighbourhood group', values='Average review rate number')

plt.figure(figsize=(10, 6))
sns.heatmap(review_rate_pivot, annot=True, fmt=".1f", cmap='coolwarm', cbar_kws={'label': 'Average Review Rate Number'})
plt.title('Heatmap of Average review rate number by cancellation policy and neighbourhood group')
plt.xlabel('Neighbourhood group')
plt.ylabel('Cancellation Policy')
plt.show()
```
![wykres8](https://github.com/user-attachments/assets/a97566e1-0e27-4a20-9cd7-3463b5632724)

The next heatmap shows us that the properties with the lowest review rates are located in Manhattan and have no clear cancellation policies and the properties with the highest review rate are located in Staten Island, that also have no clear cancellation policies.

Comparing the two heatmaps - the lowest average review rate, but also the lowest average price is achieved by properties in Manhattan that have no stated cancellation policy, the highest average review rate and one of the highest average prices is achieved by the properties located in Staten Island.

I wondered if the availability for renting during the year has any impact on the reviews and prices, so I checked the correlation between them.

```python
correlation_matrix2 = aob[['review rate number', 'availability 365', 'price']].corr()

print(correlation_matrix2)
```
|                        | Review Rate Number | Availability 365  | Price     |
|------------------------|--------------------|-------------------|-----------|
| **Review Rate Number** | 1.000000           | -0.003834         | -0.003061 |
| **Availability 365**   | -0.003834          | 1.000000          | -0.005635 |
| **Price**              | -0.003061          | -0.005635         | 1.000000  |

Based on these calculations, we see that the availability of properties during the year doesn't affect the reviews or the prices. The situation of the properties with the shortest possible length of stay is very similar: 

```python
correlation_matrix3 = aob[['review rate number', 'minimum nights', 'price']].corr()
 
print(correlation_matrix3)
```
|                      | Review Rate Number | Minimum Nights | Price     |
|----------------------|--------------------|----------------|-----------|
| **Review Rate Number** | 1.000000           | -0.002017       | -0.003061 |
| **Minimum Nights**     | -0.002017          | 1.000000        | -0.006507 |
| **Price**              | -0.003061          | -0.006507       | 1.000000  |

I've also generated graphs showing the relationship between average review rate and room type by the neighbourhood group:

```python
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_data, x='room type', y='review rate number', hue='neighbourhood group')
plt.title('Average review rate number by room type and neighbourhood group')
plt.ylabel('Average review rate number')
plt.xlabel('Room type')
plt.legend(title='Neighbourhood group',
           bbox_to_anchor=(1.05, 1),
           loc='upper left')
plt.show()
```
![wykres9](https://github.com/user-attachments/assets/4920aea8-082c-4b5c-a813-083422cb819d)

This graph shows us that the review rates for specific room types are rather even with hotel rooms slightly standing out, especially in Brooklyn.

## Conclusion


To sum up, let's answer the questions from the beginning.

### 1. Which areas are the cheapest, and which are the most expensive?

After analyzing the table, it became clear that while there are no significant differences in prices between the neighbourhood groups, they can be seen between the neighbourhoods. The lowest prices are achieved by properties in Woodrow, Staten Island ($78.00), and the highest in New Dorp, Staten Island ($757.92).

### 2. Is there a difference in rental prices between properties offering instant booking and those that don’t?

The possibility of instant booking doesn't affect the prices.

### 3. How does the cancellation policy correlate with the average rental price for similar properties? 

In most cases the cancellation policy has no effect on prices, except in Queens and Staten Island, where properties without specific rules have the highest prices. Based on the data in the table, it's difficult to clearly assess why only two neighbourhood groups have higher prices. Hovewer, two hypotheses can be posed:

- Offers without specific cancellation policies might be percieved as more exclusive - hosts may not specify them to attract guests willing to pay more.
- Hosts may deliberately set higher prices for properties without stated cancellation policy to attract guests less likely to cancel.

### 4. What factors do reviews correlate with and do they affect the final prices? 

Prices depending on the property review rate are similar. The biggest differences can be observed in Staten Island, where lower review rate is associated with the highest price of renting. based on this result, iit can be inferred that due to the higher price, guests expect a hgher standard and their disappointment results in lower reviews.

Review rates correlate clightly with cancellation policies - the lowest average review rate, as well as the lowest average price, is achiewed by properties in Manhattan, which have no stated cancellation policy, while the highest average review rate and one of the highest average prices is achieved by properties in Staten Island.

Neither the availability during the year, nor the minimum nights, affects the property's review rate or the price.

The review rates for specific room types are rather even, butr hotel rooms stand out slightly, especially in Brooklyn. This may be related to the presence of room service, which by definition isn't an option in other room types.

### 5. Do properties with higher average review scores and greater number of reviews command higher rental prices? 

The room type doesn't affect the price, it also doesn't particularly affect the review rate. As I mentioned before, only hotel rooms from Brooklyn stand out in the calculation.

## Summary

We can conclude that the presented market is a very receptive one, and practically any type of property with any kind of rental rules has a chance to find its way in. 

To make it easier for first-time hosts when determining the price of renting a property I recommend checking the prices in the specific neighbourhood they're interested in, without necassarily considering the average prices across the neighbourhood group. Deciding who the property is primarily aimed at (finding the target audience) is also they key to determine accurate rental rules and create a pricing strategy on this ground.

Since factors such as the amount of days available for renting during the year or the minimum nights doesn't affect the final prices, redirecting the attention to the standard offered might be worth considering, as well as looking at the prices of properties in the closest proximity. It's also a good practice to provide reliable information about the property and the rules of renting to avoid disappointment and result in higher ratings after the guests stay is over.
