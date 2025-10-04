import requests
import os
import json
import pandas as pd
import argparse
from tqdm import tqdm

API_KEY = os.environ['OPENROUTER_API']
print(API_KEY)

def ask_llm(s):
    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
            "Authorization": "Bearer " + API_KEY,
            "HTTP-Referer": "<your>", # Optional. Site URL for rankings on openrouter.ai.
            "X-Title": "<mom>", # Optional. Site title for rankings on openrouter.ai.
        },
    #"qwen/qwen-turbo", # Optional
        data=json.dumps({
            # "model": "qwen/qwen3-235b-a22b-thinking-2507",
            "model": "deepseek/deepseek-v3.2-exp",
            "messages": [
            {
                "role": "user",
                "content": s
            }
            ]
        })
        )
    print(response)
    response_json = response.json()
    
    print(response.json(), "\n\n\n")
    response_json = response.json()
    
    if response.status_code == 200:
        answer = ""
        if 'choices' in response_json and response_json['choices']:
            first_choice = response_json['choices'][0]
            if 'message' in first_choice and 'content' in first_choice['message']:
    
                answer = first_choice['message']['content']
    
                print(answer)
            else:
                print("Could not find 'content' in the message.")
        else:
            print("Could not find 'choices' in the response or 'choices' is empty.")
    
        return answer

def parse_ticker_data(data_string, tickers_string, cnt_day):
    """
    Parses a string of ticker data into a dictionary, prioritizing non-'nan' values
    and ensuring all tickers from tickers_string are present with cnt_day values.

    Args:
        data_string: A string where each line contains a ticker symbol
                     followed by space-separated values (numbers or 'nan').
        tickers_string: A comma-separated string of all expected ticker symbols.
        cnt_day: The expected number of values for each ticker.

    Returns:
        A dictionary where keys are ticker symbols and values are lists
        of the corresponding data values, with non-'nan' values prioritized.
        Returns a dictionary with all tickers and 'nan' values if parsing fails.
    """
    data_dict = {}
    expected_tickers = tickers_string.split(',')
    default_values = ['nan'] * cnt_day

    try:
        lines = data_string.strip().split('\n')
        for line in lines:
            parts = line.split()
            if parts:
                ticker = parts[0]
                values = parts[1:]
                # Convert numeric strings to floats, keep 'nan' as string
                processed_values = [float(v) if v.lower() != 'nan' else v for v in values]

                if ticker in data_dict:
                    # If ticker already exists, update values, prioritizing non-'nan'
                    existing_values = data_dict[ticker]
                    updated_values = []
                    for i in range(len(processed_values)):
                        if processed_values[i] != 'nan':
                            updated_values.append(processed_values[i])
                        else:
                            updated_values.append(existing_values[i])
                    data_dict[ticker] = updated_values
                else:
                    # If ticker doesn't exist, add it to the dictionary
                    data_dict[ticker] = processed_values
    except Exception as e:
        print(f"Error parsing data string: {e}")
        # Return dictionary with default values if parsing fails
        return {ticker: default_values for ticker in expected_tickers}

    # Ensure all expected tickers are in the dictionary, filling with default values if missing
    for ticker in expected_tickers:
        if ticker not in data_dict:
            data_dict[ticker] = default_values
        # Ensure each ticker has cnt_day values
        while len(data_dict[ticker]) < cnt_day:
            data_dict[ticker].append('nan')
        data_dict[ticker] = data_dict[ticker][:cnt_day] # Truncate if more than cnt_day values


    return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run baseline solution for the competition")
    parser.add_argument("--candles_path", type=str, required=True,
                       help="Path to training data (candles)")
    parser.add_argument("--news_path", type=str, required=True,
                       help="Path to training data (candles)")
    parser.add_argument("--output_path", type=str, default="submission.csv",
                       help="Path to save submission file (default: submission.csv)")
    args = parser.parse_args()
    
    df_candles = pd.read_csv(args.candles_path) 
    unique_tickers = df_candles['ticker'].unique()
    tickers_string = ','.join(unique_tickers)
    print(tickers_string)
    df_news = pd.read_csv(args.news_path)

    df_news['publish_date'] = pd.to_datetime(df_news['publish_date'])
    column_names = " ".join(df_news.columns)
    unique_dates = df_news['publish_date'].dt.date.unique()
    sorted_unique_dates = sorted(unique_dates)
    last_day = sorted_unique_dates[-1]
    cnt_day = 20
    print(cnt_day)
    print(f"Making predictions using news up to {cnt_day} days before {last_day}")

    predict_data_begin = last_day + pd.Timedelta(days=1)
    predict_data_end = predict_data_begin + pd.Timedelta(days=21)
    my_prompt1 = f'''На основе предоставленных новостей спрогнозируйте изменение цены акций в процентах для каждого из следующих тикеров на каждый из дней от текущей даты, даты на которые нужн оэто предсказать находятся в диапазоне от {predict_data_begin} до {predict_data_end}.
    Если новость не относится к конкретному тикеру, укажите 'nan'.
    Предоставьте ответ в формате: тикер, затем {20} значений изменения цены в процентах, разделенных пробелами. Каждая строка должна содержать только тикер и его значения. Не включайте никаких дополнительных пояснений или текста.
    
    Тикеры: {tickers_string}
    Новости:
    '''
    data_llm_predict = dict()
    column_names = " ".join(df_news.columns)

    # Get the column names
    column_names = " ".join(df_news.columns)

    # Iterate through each date in the range
    pbar = tqdm(total=cnt_day, desc="Running openrouter API...")
    current_date = last_day - pd.Timedelta(days=cnt_day-1)
    while current_date <= last_day:
        try:
            # print(f"\nData for date: {current_date}")
            # Print the column names
            #print(column_names)
            # Filter the DataFrame for the current date
            rows_for_date = df_news[df_news['publish_date'].dt.date == current_date]
        
            # Concatenate all columns for each row into a single string and then join those strings with a newline
            combined_text_for_date = column_names + '\n' + '\n'.join(rows_for_date.astype(str).agg(' '.join, axis=1))
            # print(combined_text_for_date)
            promt = my_prompt1 + combined_text_for_date + f'\n это новости за {current_date}'
        
            answer = ask_llm(promt)
        
            parsed_data = parse_ticker_data(answer, tickers_string, 20)
            # print("parsed_data")
            # print(parsed_data)
            data_llm_predict[current_date] = parsed_data
        
        except Exception as e:
            print("failed to perform api call", e)
            pass
        # Move to the next date
        current_date += pd.Timedelta(days=1)
        pbar.update(1) 

    print(data_llm_predict)
    output = pd.DataFrame(index=unique_tickers, columns = [f"p{i}" for i in range(1, 21)], dtype=float)
    output = output.fillna(0.)
    for date in data_llm_predict:
        print("Running oracle at date {date}")
        try:
            this_date_result = pd.DataFrame(data_llm_predict[date], dtype='float').T
            this_date_result.columns = [f"p{i}" for i in range(1, 21)]
            output += this_date_result.fillna(0.) / cnt_day
        except:
            pass
    (output / 100).rename_axis('ticker').to_csv(args.output_path)
    print("Submission file saved")

