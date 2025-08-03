from utils.indicators import calcStochRSI, calcMACD
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing

def process_file(file_path):
    data = pd.read_csv(file_path, skiprows=1)
    
    # skip raw data with too short data length
    if len(data) < 30:
        return 
    
    data = data.drop(columns=data.columns[[0, 2]]) # delete 'unix' and 'symbol' columns
    
    if 'Date' in data.columns:
        data = data.rename(columns={'Date': 'date'})
    if 'open' in data.columns:
        data = data.rename(columns={'open': 'Open'})
    if 'high' in data.columns:
        data = data.rename(columns={'high': 'High'})
    if 'low' in data.columns:
        data = data.rename(columns={'low': 'Low'})
    if 'close' in data.columns:
        data = data.rename(columns={'close': 'Close'})
    
    assert all(data.columns[:5] == ['date','Open','High','Low','Close'])
    
    data = data.sort_values(by='date') # change the order of data into time-order
    data = data.reset_index(drop=True)
    
    data['date'] = pd.to_datetime(data['date'], format="mixed").dt.strftime("%Y-%m-%d %H:%M:%S")
    
    data = calcStochRSI(data)
    data = calcMACD(data)
    
    # Delete the first several rows due to stochastic RSI nan values
    data = data.drop(range(30)) #* Should be modified according to the RSI hyperparameter
    # save preprocessed data
    file_save = './data' + file_path[len(folder_path):]
    file_save_dir = './data' + os.path.dirname(file_path[len(folder_path):])
    if not os.path.exists(file_save_dir):
        os.makedirs(file_save_dir)
    data.to_csv(file_save, index=False)
    

def main(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.csv'):
                file_paths.append(os.path.join(root, filename))
    
    # CPU multiprocessing 
    with multiprocessing.Pool(processes=64) as pool:
        results = []
        with tqdm(total=len(file_paths), desc="Processing files") as pbar:
            for file_path in file_paths:
                result = pool.apply_async(process_file, (file_path,))
                results.append(result)
            for result in results:
                result.get()
                pbar.update(1)
    
if __name__ == '__main__':
    # * raw data folder path
    folder_path = './data_raw'
    main(folder_path)