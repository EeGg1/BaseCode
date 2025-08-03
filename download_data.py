# https://www.cryptodatadownload.com/cdd/Binance_BNBBTC_1h.csv
# from binance.client import Client
# from binance.client import Client

# api_key = "3gWFQGRGs9aLkdBr1xqEZJf9xpgryDOlKK65C5u28mhxuk4qXhxBzJT4MYBeBs8B"
# api_secret = "AHOMXHUCvlQdO0omkRC7ofBCGTNH3qq0uIdfUGuXGejrLzWvDZV8sXPlck0knyf7"

# client = Client()
# exchange_info = client.get_exchange_info()
# usdt = set()
# for s in exchange_info['symbols']:
#     if s['symbol'].endswith('USDT'):
#         usdt.add(s['symbol'])
# print(usdt)
# print(len(usdt))

# from binance.spot import Spot as Client

# client = Client()
# exchange_info = client.exchange_info()
# symbol_pair = set()
# for s in exchange_info['symbols']:
#     symbol_pair.add(s['symbol'])

# # Save symbol_pair to a text file
# with open('symbol_pairs.txt', 'w') as file:
#     for symbol in symbol_pair:
#         file.write(symbol + '\n')
# print(len(symbol_pair))

import json
import requests
import os
from tqdm import tqdm
#######################################################################################
# region Binance Spot Day Data
# Make a GET request to the website
response = requests.get('https://api.cryptodatadownload.com/v1/data/ohlc/binance/available')

# Print the status code
print("status_code" + str(response.status_code))

# Print the response body
# Convert the response body to JSON
data = json.loads(response.text)["data"]

# Extract the "symbol" field from each item in the response
symbols = set()
for item in data:
    symbol = item["symbol"]
    if symbol not in symbols:
        symbols.add(symbol)

# Print the symbols
print("number of symbols:" + str(len(symbols)))

directory = "./data/crypto_binance_spot_1d"
if not os.path.exists(directory):
    os.makedirs(directory)
                
for symbol in tqdm(symbols, desc="Downloading"):
    # https://www.cryptodatadownload.com/cdd/Binance_1INCHBTC_d.csv
    url = "https://www.cryptodatadownload.com/cdd/Binance_" + symbol + "_d.csv"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open("./data/crypto_binance_spot_1d/" + symbol + ".csv", "wb") as file:
                file.write(response.content)
        else:
            raise Exception("Failed to download file")
    except Exception as e:
        print("Error downloading file for symbol:", symbol)
        print("Error message:", str(e))

print("Downloaded", len(symbols), "files")
# endregion
# #######################################################################################
# region Binance Spot Hour Data
# Make a GET request to the website
response = requests.get('https://api.cryptodatadownload.com/v1/data/ohlc/binance/guest/spot/available')

# Print the status code
print("status_code" + str(response.status_code))

# Print the response body
# Convert the response body to JSON
data = json.loads(response.text)["data"]

# Extract the "symbol" field from each item in the response
symbols = set()
for item in data:
    symbol = item["symbol"]
    if symbol not in symbols:
        symbols.add(symbol)

# Print the symbols
print("number of symbols:" + str(len(symbols)))

directory = "./data/crypto_binance_spot_1h"
if not os.path.exists(directory):
    os.makedirs(directory)
                
for symbol in tqdm(symbols, desc="Downloading"):
    url = "https://www.cryptodatadownload.com/cdd/Binance_" + symbol + "_1h.csv"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open("./data/crypto_binance_spot_1h/" + symbol + ".csv", "wb") as file:
                file.write(response.content)
        else:
            raise Exception("Failed to download file")
    except Exception as e:
        print("Error downloading file for symbol:", symbol)
        print("Error message:", str(e))

print("Downloaded", len(symbols), "files")
# endregion
#######################################################################################
# region Binance Futures Data
# Make a GET request to the website
response = requests.get('https://api.cryptodatadownload.com/v1/data/ohlc/binance/futures/um/available')

# Print the status code
print("status_code" + str(response.status_code))

# Print the response body
# Convert the response body to JSON
data = json.loads(response.text)["data"]

# Extract the "symbol" field from each item in the response
symbols = [[item["symbol"], item['timeframe']] for item in data]

# Print the symbols
print("number of symbols:" + str(len(symbols)))

directory1= "./data/crypto_binance_futures_1d"
directory2= "./data/crypto_binance_futures_1h"
if not os.path.exists(directory1):
    os.makedirs(directory1)
if not os.path.exists(directory2):
    os.makedirs(directory2)
                
for symbol in tqdm(symbols, desc="Downloading"):
    if symbol[1] == 'day':
        # https://www.cryptodatadownload.com/cdd/1000BTTCUSDT_Binance_futures_UM_day.csv
        url = "https://www.cryptodatadownload.com/cdd/" + symbol[0] + "_Binance_futures_UM_day.csv"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open("./data/crypto_binance_futures_1d/" + symbol[0] + ".csv", "wb") as file:
                    file.write(response.content)
            else:
                raise Exception("Failed to download file")
        except Exception as e:
            print("Error downloading file for symbol:", symbol)
            print("Error message:", str(e))
    elif symbol[1] == 'hour':
        # https://www.cryptodatadownload.com/cdd/1000BTTCUSDT_Binance_futures_UM_hour.csv
        url = "https://www.cryptodatadownload.com/cdd/" + symbol[0] + "_Binance_futures_UM_hour.csv"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open("./data/crypto_binance_futures_1h/" + symbol[0] + ".csv", "wb") as file:
                    file.write(response.content)
            else:
                raise Exception("Failed to download file")
        except Exception as e:
            print("Error downloading file for symbol:", symbol)
            print("Error message:", str(e))
    
print("Downloaded", len(symbols), "files")
# endregion