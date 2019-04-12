# make timestamps to datetime format
import datetime
import os
import sys

dataset = sys.argv[1]
path = '../data/' + dataset
os.makedirs(dataset + '_TA', exist_ok=True)
newpath = 'data/' + dataset + '_TA/'
filelist = os.listdir(path)
for filename in filelist:
    if filename.startswith("stat"):
        continue

    fp = open(os.path.join(path, filename))
    fw = open(os.path.join(newpath, filename), "w")

    start_time_str = "2018-01-01"
    format = "%Y-%m-%d"
    start_time = datetime.datetime.strptime(start_time_str, format)

    time_id = 3

    for i, line in enumerate(fp):
        info = line.strip().split("\t")

        time = start_time + datetime.timedelta(hours=int(info[time_id]))
        time_str = time.strftime(format)

        fw.write("%-5d\t%-5d\t%-3d\t%-s\t0\n" % (int(info[0]), int(info[1]), int(info[2]), time_str))

    fp.close()
    fw.close()
