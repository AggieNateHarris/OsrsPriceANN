import requests
import csv
import time
from time import sleep
import pandas as pd
import json
import numpy as np
from os.path import exists
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np
import matplotlib.pyplot as plt

# *************************************************************
# /////////////////////////////////////////////////////////////

MAPPINGURL = "https://prices.runescape.wiki/api/v1/osrs/mapping"
LATESTPRICESURL = "https://prices.runescape.wiki/api/v1/osrs/latest"
# must set timestamp, time since epoch
TIMESTAMPPRICES1H = "https://prices.runescape.wiki/api/v1/osrs/1h?timestamp="
ITEMTIMESERIES = "https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep="
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
    # possible to put custom time in for epoch_time if wished
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
            oneHot = pd.get_dummies(workSpaceDF[0]['name'])
            workSpaceDF[0].drop('name', axis=1)
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

def getItemPrice(itemId, numOfTimePoints, interval=21600):
    # from right now
    epoch_time = int((time.time() / SIXH))
    # cast epoch time to string, use above or a custom time point
    epoch_time = str(1629990000 - interval) #str(epoch_time * SIXH)
    namesMapping = pd.read_csv('MappingData.csv')
    index = namesMapping.index[namesMapping['id'] == itemId].tolist()
    itemName = ""
    if (len(index) > 0):
        itemName = namesMapping.loc[namesMapping['id'] == itemId]['name'][index[0]]
    else:
        itemName = None
    finalDf = None
    for i in range(numOfTimePoints):
        r = requests.get(TIMESTAMPPRICES1H+epoch_time, headers=headers)
        print("Currently " + str(i) + "/" + str(numOfTimePoints))

        # parse fields from JSON response
        js = json.loads(r.content)
        df = pd.DataFrame(js['data'])

        tempDf = df.get(str(itemId))
        if tempDf is not None:
            tempDf = df[[str(itemId)]]
            tempDf = tempDf.T
            tempDf['timestamp'] = epoch_time
            if finalDf is None:
                finalDf = tempDf
            else:
                tempDf = [finalDf, tempDf]
                finalDf = pd.concat(tempDf)

        epoch_time = str(int(epoch_time) - interval)
        sleep(10)

    finalDf['name'] = itemName
    finalDf = finalDf.reindex(columns=['timestamp', 'avgHighPrice', 'avgLowPrice', 'highPriceVolume', 'lowPriceVolume', 'name'])
    # finalDf = finalDf.T
    finalDf.to_csv(itemName + '3.csv', index=False)

def getTimeSeries(itemId, timestep='6h'):
    r = requests.get(ITEMTIMESERIES + timestep + '&id=' + str(itemId), headers=headers)

    # parse fields from JSON response
    js = json.loads(r.content)
    df = pd.DataFrame(js['data'])

    namesMapping = pd.read_csv('MappingData.csv')
    index = namesMapping.index[namesMapping['id'] == itemId].tolist()
    itemName = ""
    if (len(index) > 0):
        itemName = namesMapping.loc[namesMapping['id'] == itemId]['name'][index[0]]
    else:
        itemName = None

    df['name'] = itemName

    df.to_csv(itemName + 'TimeSeries.csv', index=False)

def combineFiles(listOfFileNames):
    num = 0
    df = None
    for name in listOfFileNames:
        if df is None:
            df = pd.read_csv(name)
        else:
            temp = pd.read_csv(name)
            df = pd.concat([df, temp], ignore_index=True)
    return df


def saveToCSV(listOfItemData):
    with open('itemHistory', 'a') as f:
        write = csv.writer(f)
        write.writerow(itemFields)
        write.writerows(listOfItemData)


def createLabeledData(listOfFileNames):
    df = combineFiles(listOfFileNames)

    # upcoming update context, manually set
    df['update'] = 0
    df['update1'] = 0
    df['update2'] = 0
    df['update3'] = 1

    df.dropna()

    priceIn6H = []
    sixHourPrice = []
    twelveHourPrice = []
    eighteenHourPrice = []
    twentyFourHourPrice = []
    thirtyHourPrice = []
    thirtySixHourPrice = []
    fortyTwoHourPrice = []
    fortyEightHourPrice = []
    fiftyFourHourPrice = []
    seventyTwoHourPrice = []
    ninteySixHourPrice = []
    hundredTwentyHourPrice = []
    hundredFortyFourHourPrice = []

    sixHourVolume = []
    twelveHourVolume = []
    eighteenHourVolume = []
    twentyFourHourVolume = []
    thirtyHourVolume = []
    thirtySixHourVolume = []
    fortyTwoHourVolume = []
    fortyEightHourVolume = []
    fiftyFourHourVolume = []
    seventyTwoHourVolume = []
    ninteySixHourVolume = []
    hundredTwentyHourVolume = []
    hundredFortyFourHourVolume = []

    for i in range(df.shape[0]):
        if i - 1 >= 0:
            priceIn6H.append(df['avgHighPrice'][i - 1])
        if i + 1 < df.shape[0]:
            sixHourPrice.append(df['avgHighPrice'][i + 1])
            sixHourVolume.append(df['highPriceVolume'][i + 1])
        if i + 2 < df.shape[0]:
            twelveHourPrice.append(df['avgHighPrice'][i + 2])
            twelveHourVolume.append(df['highPriceVolume'][i + 2])
        if i + 3 < df.shape[0]:
            eighteenHourPrice.append(df['avgHighPrice'][i + 3])
            eighteenHourVolume.append(df['highPriceVolume'][i + 3])
        if i + 4 < df.shape[0]:
            twentyFourHourPrice.append(df['avgHighPrice'][i + 4])
            twentyFourHourVolume.append(df['highPriceVolume'][i + 4])
        if i + 5 < df.shape[0]:
            thirtyHourPrice.append(df['avgHighPrice'][i + 5])
            thirtyHourVolume.append(df['highPriceVolume'][i + 5])
        if i + 6 < df.shape[0]:
            thirtySixHourPrice.append(df['avgHighPrice'][i + 6])
            thirtySixHourVolume.append(df['highPriceVolume'][i + 6])
        if i + 7 < df.shape[0]:
            fortyTwoHourPrice.append(df['avgHighPrice'][i + 7])
            fortyTwoHourVolume.append(df['highPriceVolume'][i + 7])
        if i + 8 < df.shape[0]:
            fortyEightHourPrice.append(df['avgHighPrice'][i + 8])
            fortyEightHourVolume.append(df['highPriceVolume'][i + 8])
        if i + 9 < df.shape[0]:
            fiftyFourHourPrice.append(df['avgHighPrice'][i + 9])
            fiftyFourHourVolume.append(df['highPriceVolume'][i + 9])
        if i + 12 < df.shape[0]:
            seventyTwoHourPrice.append(df['avgHighPrice'][i + 12])
            seventyTwoHourVolume.append(df['highPriceVolume'][i + 12])
        if i + 16 < df.shape[0]:
            ninteySixHourPrice.append(df['avgHighPrice'][i + 16])
            ninteySixHourVolume.append(df['highPriceVolume'][i + 16])
        if i + 20 < df.shape[0]:
            hundredTwentyHourPrice.append(df['avgHighPrice'][i + 20])
            hundredTwentyHourVolume.append(df['highPriceVolume'][i + 20])
        if i + 24 < df.shape[0]:
            hundredFortyFourHourPrice.append(df['avgHighPrice'][i + 24])
            hundredFortyFourHourVolume.append(df['highPriceVolume'][i + 24])

    df = df.drop(0, axis=0)
    df['priceIn6H'] = priceIn6H
    df['Price6H'] = sixHourPrice
    df['Volume6H'] = sixHourVolume
    df = df.drop(df.shape[0] - 1, axis=0)
    df['Price12H'] = twelveHourPrice
    df['Volume12H'] = twelveHourVolume
    df = df.drop(df.shape[0] - 1, axis=0)
    df['Price18H'] = eighteenHourPrice
    df['Volume18H'] = eighteenHourVolume
    df = df.drop(df.shape[0] - 1, axis=0)
    df['Price24H'] = twentyFourHourPrice
    df['Volume24H'] = twentyFourHourVolume
    df = df.drop(df.shape[0] - 1, axis=0)
    df['Price30H'] = thirtyHourPrice
    df['Volume30H'] = thirtyHourVolume
    df = df.drop(df.shape[0] - 1, axis=0)
    df['Price36H'] = thirtySixHourPrice
    df['Volume36H'] = thirtySixHourVolume
    df = df.drop(df.shape[0] - 1, axis=0)
    df['Price42H'] = fortyTwoHourPrice
    df['Volume42H'] = fortyTwoHourVolume
    df = df.drop(df.shape[0] - 1, axis=0)
    df['Price48H'] = fortyEightHourPrice
    df['Volume48H'] = fortyEightHourVolume
    df = df.drop(df.shape[0] - 1, axis=0)
    df['Price54H'] = fiftyFourHourPrice
    df['Volume54H'] = fiftyFourHourVolume
    df = df.drop([df.shape[0] - 1, df.shape[0] - 2, df.shape[0] - 3], axis=0)
    df['Price72H'] = seventyTwoHourPrice
    df['Volume72H'] = seventyTwoHourVolume
    df = df.drop([df.shape[0] - 1, df.shape[0] - 2, df.shape[0] - 3, df.shape[0] - 4], axis=0)
    df['Price96H'] = ninteySixHourPrice
    df['Volume96H'] = ninteySixHourVolume
    df = df.drop([df.shape[0] - 1, df.shape[0] - 2, df.shape[0] - 3, df.shape[0] - 4], axis=0)
    df['Price120H'] = hundredTwentyHourPrice
    df['Volume120H'] = hundredTwentyHourVolume
    df = df.drop([df.shape[0] - 1, df.shape[0] - 2, df.shape[0] - 3, df.shape[0] - 4], axis=0)
    df['Price144H'] = hundredFortyFourHourPrice
    df['Volume144H'] = hundredFortyFourHourVolume

    itemName = df['name'][1]
    oneHot = pd.get_dummies(df['name'])
    df.drop('name', axis=1)
    df = df.join(oneHot)
    df.to_csv(itemName + 'Rdy.csv', index=False)


# input: valid_X is the validation set of X (datapoints where ground truth field is removed)
#        valid_Y is ground truth field corresponding to valid_X validation set
#        test_prediction is the list of predicted Y values for valid_X validation set
#        test_prediction is predicted values to be compared with true Y values valid_Y
#        errMargin is the allowed error margin to still be counted correct, 0.01 = 1%
def score(valid_X, valid_Y, test_prediction, errMargin=0.01):
    success = 0
    directionSuccess = 0
    i = 0
    valList = []
    for val in valid_X['avgHighPrice']:
        valList.append(val)
    for y in valid_Y:
        if abs(test_prediction[i] - y) < errMargin * y:
            success += 1
        if (test_prediction[i] - valList[i]) >= 0:
            if (y - valList[i]) >= 0:
                directionSuccess += 1
        if (test_prediction[i] - valList[i]) < 0:
            if (y - valList[i]) < 0:
                directionSuccess += 1
        i += 1
    print("Finished testing, results are: ")
    print("Price within margin accuracy: {0:.4f}" .format(success / len(test_prediction)))
    print("Rise/drop directional accuracy: {0:.4f}" .format(directionSuccess / len(test_prediction)))
    predFrame = pd.DataFrame(test_prediction, valid_X['timestamp'])
    groundTruthFrame = pd.DataFrame(valid_Y.values, valid_X['timestamp'])
    plt.title('Prediction & Ground Truth Values')
    plt.xlabel('Timestamp')
    plt.ylabel('Gp Price')
    plt.plot(predFrame, color="Green")
    plt.plot(groundTruthFrame, color="Blue")
    plt.legend(['Predictions', 'Ground Truth'])
    plt.show()


def main():
    # UNCOMMENT SECTIONS AS NEEDED
    # TODO: clean up data and update methods that ensure fields such as
    #       price6H are actually prices from 6 hours prior, there may be
    #       cases currently where there was no data point 6 hours prior
    #       so the previous data point was actually 12 hours prior and
    #       is being treated as 6h prior, this will throw off everything

    #------------------------------------------------------------------------------------------
    # 1 time thing, only need to do for new items or id:item system change at wiki prices site
    # mapping id numbers to item names
    # mapJsonToItems(MAPPINGURL)
    # ------------------------------------------------------------------------------------------


    #------------------------------------------------------------------------------------------
    # getting historic price data of ALL ITEMS
    # input is the number of points to get
    # to add to a current base of points, check oldest date of base
    # then change the epoch_time to previous point before oldest
    # TODO: automate this as inputs instead of requiring editing function/checking base
    # getHistoricPrices(40)
    # ------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------
    # get specific item price data for a number of points
    # input is the item id and the number of points
    # to add to a current base of points, check oldest date of base
    # then change the epoch_time to previous point before oldest
    # TODO: automate this as inputs instead of requiring editing function/checking base
    getItemPrice(4151, 600)
    # ------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------
    # get time series(300) for a specific item
    # currently api has a static 300 time series for last 300
    # I used this to more quickly get the most recent 300 points
    # and then getItemPrice for points beyond the 300
    # function input is the item id
    # getTimeSeries(561)
    # ------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------
    # create correctly formatted csv file for training and testing from existing data csv files
    # input is a list of csv filenames, files created from
    # previous data gathering functions work here
    # createLabeledData(['Abyssal whip.csv', 'Abyssal whip2.csv', 'Abyssal whipTimeSeries.csv'])
    # ------------------------------------------------------------------------------------------


    # # ------------------------------------------------------------------------------------------
    # # ----------------------LOADING TRAINING VALIDATION MODEL-----------------------------------
    # # ------------------------------------------------------------------------------------------
    # # initial opening and reading method for all item csv files
    # # training_set = pd.read_csv('itemsHistory0.csv')
    # # fileNum = 1
    # # while exists('itemsHistory' + str(fileNum) + '.csv'):
    # #     temp_set = pd.read_csv('itemsHistory' + str(fileNum) + '.csv')
    # #     data = [training_set, temp_set]
    # #     training_set = pd.concat(data)
    # #     fileNum += 1
    # # ------------------------------------------------------------------------------------------
    #
    # # ------------------------------------------------------------------------------------------
    # # new method for reading in prepared csv single item csv files
    # training_set = pd.read_csv('Abyssal whipRdy.csv')
    # # ------------------------------------------------------------------------------------------
    #
    # # ------------------------------------------------------------------------------------------
    # # clean up data by dropping NA data points
    # training_set = training_set.dropna()
    # print("Training set shape: ")
    # print(training_set.shape)
    # print(training_set.head())
    # # ------------------------------------------------------------------------------------------
    #
    # # ------------------------------------------------------------------------------------------
    # # split data into train and validation sets
    # train, valid = train_test_split(training_set, test_size=0.2, random_state=42, shuffle=False)
    # print("Training subset after train-valid split: ")
    # print(train.shape)
    # print("Valid subset after train-valid split: ")
    # print(valid.shape)
    # # ------------------------------------------------------------------------------------------
    #
    # # ------------------------------------------------------------------------------------------
    # # create or choose an existing model that will be used for training/validation
    # # working_model = RandomForestClassifier(n_estimators=100, max_depth=80, random_state=1)
    # # working_model = RandomForestRegressor(n_estimators=100, max_depth=80, random_state=1)
    # working_model = AdaBoostRegressor()
    # # ------------------------------------------------------------------------------------------
    #
    # # ------------------------------------------------------------------------------------------
    # # drop ground truth in training and validation X
    # # either drop or 1-hot-encode item name
    # # set train and valid Y to their respective ground truth
    # # TODO: More feature selection analysis and implementation
    # train_X = train.drop('priceIn6H', axis=1)
    # train_X = train_X.drop('name', axis=1)
    # valid_X = valid.drop('priceIn6H', axis=1)
    # valid_X = valid_X.drop('name', axis=1)
    # train_Y = train['priceIn6H']
    # valid_Y = valid['priceIn6H']
    # # fill any NA not dropped as 0
    # train_X = train_X.fillna(0)
    # valid_X = valid_X.fillna(0)
    # print("train_X head")
    # print(train_X.head())
    #
    # print("Training subset after train-valid split & feature selection: ")
    # print(train_X.shape)
    # print(train_Y.shape)
    # print("Valid subset after train-valid split & feature selection: ")
    # print(valid_X.shape)
    # print(valid_Y.shape)
    # # ------------------------------------------------------------------------------------------
    #
    # # ------------------------------------------------------------------------------------------
    # # Model training
    # working_model.fit(train_X, train_Y)
    # # ------------------------------------------------------------------------------------------
    #
    # # ------------------------------------------------------------------------------------------
    # # if desired, load a previous model to be used
    # # with open('best_model.pkl', 'rb') as f:
    # #     loaded_model = pickle.load(f)
    # # test_prediction = loaded_model.predict(valid_X)
    # # ------------------------------------------------------------------------------------------
    #
    # # ------------------------------------------------------------------------------------------
    # # set working model to the loaded model if not training
    # # working_model = loaded_model
    # # ------------------------------------------------------------------------------------------
    #
    # # ------------------------------------------------------------------------------------------
    # # given a set of validation X and your trained model,
    # # produce Y predictions
    # test_prediction = working_model.predict(valid_X)
    # # ------------------------------------------------------------------------------------------
    #
    # # ------------------------------------------------------------------------------------------
    # # base built in scoring method, for this application
    # # this method isn't very good because we don't need to
    # # be exactly correct down to a single gp, something like
    # # within 1% or 0.1% of the price is fine
    # # predictions = working_model.score(valid_X, valid_Y)
    # # print("Finished training and testing: ")
    # # print(predictions)
    # # ------------------------------------------------------------------------------------------
    #
    # # ------------------------------------------------------------------------------------------
    # # give an accuracy score for validation set
    # # testing predicted Y values against ground truth valid_Y set
    # # that corresponds to valid_X validation set
    # # input is validation X set, validation Y set, predicted Y set
    # # and an errMargin allowed, eg 0.01 means predicted price
    # # must be within 1% of the actual price to be counted as true
    # score(valid_X, valid_Y, test_prediction, errMargin=0.01);
    # # ------------------------------------------------------------------------------------------
    #
    # # ------------------------------------------------------------------------------------------
    # # Save the model if desired
    # val = input("Do you want to save the model? Y/N")
    # if val == "Y":
    #     with open('best_model.pkl', 'wb') as f:
    #         pickle.dump(working_model, f)
    # # ------------------------------------------------------------------------------------------
    # # ---------------------- END LOADING TRAINING VALIDATION MODEL------------------------------
    # # ------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()