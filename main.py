import requests
import csv
import time
from time import sleep
import pandas as pd
import json
from os.path import exists
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

# *************************************************************
# /////////////////////////////////////////////////////////////

MAPPINGURL = "https://prices.runescape.wiki/api/v1/osrs/mapping"
LATESTPRICESURL = "https://prices.runescape.wiki/api/v1/osrs/latest"
# must set timestamp, time since epoch
TIMESTAMPPRICES1H = "https://prices.runescape.wiki/api/v1/osrs/1h?timestamp="
SIXH = 6 * 3600

# PriceXH are price differences, diff between XH time avgHighPrice & avgLowPrice
# diff between avgHighPrice & avgLowPrice at initializing point is held at 6H, 12H - 6 hours before
# VolumeXH is the same format but for volume
# update fields are bools, 0 false, 1 true, 'update' less than a month before OR after BIG update
# 'update1' is (2-1] months only before BIG update, 'update2' (3-2], 'update3' 3+
# priceIn6Hs is the field that the NN will try to predict, and therefore also the ground truth when labeled
itemFields = ['Name', 'avgHighPrice', 'avgLowPrice', 'highPriceVolume', 'lowPriceVolume',
              'update', 'update1', 'update2', 'update3', 'priceIn6H'
              'Price6H', 'Price12H', 'Price18H', 'Price24H', 'Price30H', 'Price36H', 'Price42H',
              'Price48H', 'Price54H', 'Price72H', 'Price96H', 'Price120H', 'Price144H',
              'Volume6H', 'Volume12H', 'Volume18H', 'Volume24H', 'Volume30H', 'Volume36H', 'Volume42H',
              'Volume48H', 'Volume54H', 'Volume72H', 'Volume96H', 'Volume120H', 'Volume144H']

headers = {
    'User-Agent' : 'Item Info',
    'From' : 'granitepure10@gmail.com'
}

# 1 time thing, get json that has ids & item names, remove extra info
# and save to csv the ids with item names
def mapJsonToItems(url):
    r = requests.get(url, headers=headers)
    js = json.loads(r.content)
    df = pd.DataFrame(js)
    data = [df['id'], df['name']]
    headers2 = ['id', 'name']
    newDF = pd.concat(data, axis=1, keys=headers2)
    newDF.to_csv(path_or_buf='MappingData.csv', index=False)

def getHistoricPrices(numOfTimePoints, interval=21600):
    # starting time will always be based on recent 6H multiple from epoch
    # instead of this exact second - to keep everything lined up
    epoch_time = int((time.time() / SIXH))
    epoch_time = str(epoch_time * SIXH)
    namesMapping = pd.read_csv('MappingData.csv')
    workSpaceDF = []
    fileNum = 0
    for i in range(numOfTimePoints):
        r = requests.get(TIMESTAMPPRICES1H+epoch_time, headers=headers)

        # parse fields from JSON response
        js = json.loads(r.content)
        df = pd.DataFrame(js['data'])
        df = df.T

        # get name field from id - name map
        names = []
        for df_id in df.index:
            val = int(df_id)
            index = namesMapping.index[namesMapping['id'] == val].tolist()
            if (len(index) > 0):
                names.append(namesMapping.loc[namesMapping['id'] == val]['name'][index[0]])
            else:
                names.append(None)
        df['Name'] = names

        # upcoming update context, manually set
        df['update'] = 0
        df['update1'] = 0
        df['update2'] = 0
        df['update3'] = 1

        df.dropna()
        workSpaceDF.append(df)

        if len(workSpaceDF) >= 27:
            oneHot = pd.get_dummies(workSpaceDF[0]['Name'])
            workSpaceDF[0].drop('Name', axis=1)
            workSpaceDF[0] = workSpaceDF[0].join(oneHot)
            workSpaceDF[0].to_csv('itemsHistory' + str(fileNum) + '.csv', index=False)
            fileNum += 1
            del workSpaceDF[0]

        # if there's 2+ dataframes collected, begin
        if len(workSpaceDF) > 1:
            # set newest df's future price to previous df price
            workSpaceDF[-1]['priceIn6H'] = workSpaceDF[-2]['avgHighPrice']
            # for each df that still needs updates, update with new available df info
            for dfIndex in range(len(workSpaceDF)):
                # df 1 from end of list
                if len(workSpaceDF) - 1 - dfIndex == 1:
                    workSpaceDF[dfIndex]['Price6H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume6H'] = workSpaceDF[-1]['highPriceVolume']
                # df 2 from end of list, and so on...
                elif len(workSpaceDF) - 1 - dfIndex == 2:
                    workSpaceDF[dfIndex]['Price12H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume12H'] = workSpaceDF[-1]['highPriceVolume']
                elif len(workSpaceDF) - 1 - dfIndex == 3:
                    workSpaceDF[dfIndex]['Price18H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume18H'] = workSpaceDF[-1]['highPriceVolume']
                elif len(workSpaceDF) - 1 - dfIndex == 4:
                    workSpaceDF[dfIndex]['Price24H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume24H'] = workSpaceDF[-1]['highPriceVolume']
                elif len(workSpaceDF) - 1 - dfIndex == 5:
                    workSpaceDF[dfIndex]['Price30H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume30H'] = workSpaceDF[-1]['highPriceVolume']
                elif len(workSpaceDF) - 1 - dfIndex == 6:
                    workSpaceDF[dfIndex]['Price36H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume36H'] = workSpaceDF[-1]['highPriceVolume']
                elif len(workSpaceDF) - 1 - dfIndex == 7:
                    workSpaceDF[dfIndex]['Price42H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume42H'] = workSpaceDF[-1]['highPriceVolume']
                elif len(workSpaceDF) - 1 - dfIndex == 8:
                    workSpaceDF[dfIndex]['Price48H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume48H'] = workSpaceDF[-1]['highPriceVolume']
                elif len(workSpaceDF) - 1 - dfIndex == 9:
                    workSpaceDF[dfIndex]['Price54H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume54H'] = workSpaceDF[-1]['highPriceVolume']
                elif len(workSpaceDF) - 1 - dfIndex == 12:
                    workSpaceDF[dfIndex]['Price72H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume72H'] = workSpaceDF[-1]['highPriceVolume']
                elif len(workSpaceDF) - 1 - dfIndex == 16:
                    workSpaceDF[dfIndex]['Price96H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume96H'] = workSpaceDF[-1]['highPriceVolume']
                elif len(workSpaceDF) - 1 - dfIndex == 20:
                    workSpaceDF[dfIndex]['Price120H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume120H'] = workSpaceDF[-1]['highPriceVolume']
                elif len(workSpaceDF) - 1 - dfIndex == 24:
                    workSpaceDF[dfIndex]['Price144H'] = workSpaceDF[-1]['avgHighPrice']
                    workSpaceDF[dfIndex]['Volume144H'] = workSpaceDF[-1]['highPriceVolume']

        epoch_time = str(int(epoch_time) - interval)
        sleep(15)


def saveToCSV(listOfItemData):
    with open('itemHistory', 'a') as f:
        write = csv.writer(f)
        write.writerow(itemFields)
        write.writerows(listOfItemData)


def main():
    # mapping id numbers to item names
    # mapJsonToItems(MAPPINGURL)

    # # getting historic price data
    # getHistoricPrices(40)

    # train and test a model on collected data
    training_set = pd.read_csv('itemsHistory0.csv')
    fileNum = 1
    while exists('itemsHistory' + str(fileNum) + '.csv'):
        temp_set = pd.read_csv('itemsHistory' + str(fileNum) + '.csv')
        data = [training_set, temp_set]
        training_set = pd.concat(data)
        fileNum += 1

    training_set = training_set.dropna()
    print("Training set shape: ")
    print(training_set.shape)
    print(training_set.head())

    train, valid = train_test_split(training_set, test_size=0.2, random_state=42, shuffle=False)
    print("Training subset after train-valid split: ")
    print(train.shape)
    print("Valid subset after train-valid split: ")
    print(valid.shape)

    working_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)

    train_X = train.drop('priceIn6H', axis=1)
    train_X = train_X.drop('Name', axis=1)
    valid_X = valid.drop('priceIn6H', axis=1)
    valid_X = valid_X.drop('Name', axis=1)
    train_Y = train['priceIn6H']
    valid_Y = valid['priceIn6H']
    train_X = train_X.fillna(0)
    valid_X = valid_X.fillna(0)
    print("train_X head")
    print(train_X.head())

    print("Training subset after train-valid split & feature selection: ")
    print(train_X.shape)
    print(train_Y.shape)
    print("Valid subset after train-valid split & feature selection: ")
    print(valid_X.shape)
    print(valid_Y.shape)

    # Model training and validation
    working_model.fit(train_X, train_Y)

    # # set working model to the loaded model if not training
    # working_model = loaded_model

    # predictions = working_model.score(valid_X, valid_Y)
    # print("Finished training and testing: ")
    # print(predictions)

    test_prediction = working_model.predict(valid_X)
    success = 0
    i = 0
    valList = []
    for val in valid_X['avgHighPrice']:
        valList.append(val)
    for y in valid_Y:
        if abs(test_prediction[i] - y) < 0.01 * valList[i]:
            success += 1
        i += 1
    print("Finished testing, results are: ")
    print(success / len(test_prediction))

    # Save the model
    val = input("Do you want to save the model? Y/N")
    if val == "Y":
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(working_model, f)

if __name__ == "__main__":
    main()