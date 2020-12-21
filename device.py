import pandas as pd
import datetime
from dateutil.parser import parse
import numpy as np
import csv

if __name__ == '__main__':
    print("test")

    df = pd.read_csv('/Users/apple4u/Desktop/goksel tez/r4.2/device.csv')
    # df.info()
    print(df.head())
    users = df.user.unique()
    print(users)
    pcs = df.pc.values
    dates = df.date.values
    activities = df.activity.values

    dict = {}

    # usr = 'BRB0355'
    # if usr == 'BRB0355':
    for usr in users:
        try:
            #print("finding device use time for user: " + usr)
            df_usr = df[df['user'] == usr]
            activities_usr = df_usr.activity.values

            #print(usr + " initially has " + str(len(df_usr)) + " records")

            aincr = 1
            delete_counter = 0
            while aincr < len(activities_usr):
                if str(activities_usr[aincr]) == str(activities_usr[aincr - 1]):
                    # print("issue with user " + usr + " " + activities_usr[aincr])
                    # print("duplicate at location " + str(aincr))
                    df_usr = df_usr.drop(df_usr.index[aincr - 1])
                    # print("dropped row " + str(aincr-1))
                    delete_counter += 1
                aincr += 1

            #print("deleted  " + str(delete_counter) + " rows from user " + usr)
            #print(usr + " now has " + str(len(df_usr)) + " records")

            # aincrd = 1
            # while aincrd < len(activities_usr) - 1:
            #      if activities_usr[aincrd] == activities_usr[aincrd - 1]:
            #          print("issue with user " + usr + " " + activities_usr[aincrd])
            #          print("duplicate at location " + str(aincrd))
            #      aincr += 1
            #      print("scanned user " + usr)

            dates_usr = df_usr.date.values

            count = 1
            timediff_usr_seconds = 0
            sum_timediff_usr_seconds = 0
            i = 0
            #print(len(dates_usr))
            while i < ((len(dates_usr) / 2) ):
                # print(dates_moh[i])
                # print(datetime.datetime.strptime(dates_moh[i], '%m/%d/%Y, %H:%M:%S'))
                # timediff_usr.append(i)
                timediff_usr = parse(dates_usr[count]) - parse(dates_usr[count - 1])
                timediff_usr_seconds = timediff_usr.seconds
                # print(i, "/", count)
                # print(timediff_usr, "   ", timediff_usr_seconds)
                sum_timediff_usr_seconds += timediff_usr_seconds
                # timediff_usr.append((parse(dates_usr[count])-parse(dates_usr[count-1])).seconds)
                count += 2
                i += 1
            average_time_diff_usr_seconds = sum_timediff_usr_seconds / len(dates_usr)
            print("average device use time for user: " + usr + " is: " + str(average_time_diff_usr_seconds))
            dict[usr] = average_time_diff_usr_seconds
        except:
            print("exception occured for user " + usr)
    print(dict)

    with open('mycsvfile.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(dict.items())