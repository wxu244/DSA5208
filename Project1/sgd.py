import numpy as np
import pandas as pd
from mpi4py import MPI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io

def split_file(comm, rank, size):
    try:
        fp = MPI.File.Open(comm, 'nytaxi2022.csv', MPI.MODE_RDONLY)
    except MPI.Exception as e:
        if rank == 0:
            print(f"Error opening file: {e}")
        return None
    file_size = fp.Get_size()
    chunk_size = file_size // size
    start_offset = rank * chunk_size
    end_offset = (rank + 1) * chunk_size
    if rank == size - 1:
        end_offset = file_size

    real_start = start_offset
    if rank != 0:
        real_start = find_new_line(fp, start_offset, file_size)

    real_end = end_offset
    if rank != size - 1:
        real_end = find_new_line(fp, end_offset, file_size)

    local_data_size = real_end - real_start
    if local_data_size < 0:
        local_data_size = 0

    local_buffer = bytearray(local_data_size)
    fp.Read_at(real_start, local_buffer)
    fp.Close()

    return local_buffer

def find_new_line(fp, pos, file_size, buffer_size = 1024):
    offset = pos
    real_offset = pos
    found_newline = False
    while not found_newline and offset < file_size:
        read_buffer = bytearray(buffer_size)
        fp.Read_at(offset, read_buffer)

        try:
            newline_pos = read_buffer.index(b'\n')
            real_offset = offset + newline_pos + 1
            found_newline = True
        except ValueError:
            offset += buffer_size
    return real_offset


def load_and_preprocess_data(buffer, rank):
    columns = [
        'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count',
        'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID',
        'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
        'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'airport_fee'
    ]
    columns_to_keep = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'RatecodeID',
                   'PULocationID', 'DOLocationID', 'payment_type', 'extra', 'total_amount']
    format_string = '%m/%d/%Y %I:%M:%S %p'

    if rank == 0:
        df = pd.read_csv(io.BytesIO(buffer),
                     header = None,
                     names = columns,
                     usecols = columns_to_keep,
                     skiprows = 1)
    else:
        df = pd.read_csv(io.BytesIO(buffer),
                     header = None,
                     names = columns,
                     usecols = columns_to_keep)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format=format_string)
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format=format_string)
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df['passenger_count'] = df['passenger_count'].astype('int64')
    df['payment_type'] = df['payment_type'].astype('int64')
    df = df[df['payment_type'] < 3]
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 6)]
    df = df[df['trip_distance'] > 0]
    df = df[df['trip_duration'] > 0]
    return df

def global_fit_and_transform(df, rank):
    features = ['passenger_count', 'trip_distance', 'RatecodeID',
                'PULocationID', 'DOLocationID', 'payment_type', 'extra', 'trip_duration']
    target = 'total_amount'

    X = df[features].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=(42 + rank))

    local_sum = X_train.sum(axis=0).astype(np.float64)
    local_sq_sum = np.power(X_train, 2).sum(axis=0).astype(np.float64)
    local_n = np.int64(X_train.shape[0])

    global_sum = np.zeros_like(local_sum)
    global_sq_sum = np.zeros_like(local_sq_sum)

    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    comm.Allreduce(local_sq_sum, global_sq_sum, op=MPI.SUM)
    global_n = comm.allreduce(local_n, op=MPI.SUM)

    mean = global_sum / global_n
    var = (global_sq_sum / global_n) - (mean ** 2)
    var[var < 1e-12] = 1e-12
    std = np.sqrt(var)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    print(std)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # todo test on local machine, set 1000 to read files
    buff = split_file(comm, rank, 1000)
    data = load_and_preprocess_data(buff, rank)
    fit = global_fit_and_transform(data, rank)