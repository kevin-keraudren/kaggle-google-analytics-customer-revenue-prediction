import os


def split_file(f_in, folder):
    os.makedirs(folder, exist_ok=True)
    with open(f_in, "r") as input_file:
        header = None
        previous_date = None
        f = None
        for line in input_file:
            if header is None:
                header = line
            else:
                date = line.split(',')[1]
                if previous_date is None or date != previous_date:
                    if f is not None:
                        f.close()
                    previous_date = date
                    f = open(os.path.join(folder, "%s.csv" % date), "w")
                    f.write(header)
                    f.write(line)
                else:
                    f.write(line)


split_file("train.csv", "train")
split_file("test.csv", "test")
