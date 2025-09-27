import argparse
import time

import numpy as np
import pandas as pd
from mpi4py import MPI
from sklearn.model_selection import train_test_split
import io

from nn import NN


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


def find_new_line(fp, pos, file_size, buffer_size=1024):
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
    columns_to_keep = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance',
                       'RatecodeID',
                       'PULocationID', 'DOLocationID', 'payment_type', 'extra', 'total_amount']
    format_string = '%m/%d/%Y %I:%M:%S %p'

    if rank == 0:
        df = pd.read_csv(io.BytesIO(buffer),
                         header=None,
                         names=columns,
                         usecols=columns_to_keep,
                         skiprows=1)
    else:
        df = pd.read_csv(io.BytesIO(buffer),
                         header=None,
                         names=columns,
                         usecols=columns_to_keep)

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
    return df, len(columns_to_keep)


def global_fit_and_transform(df, comm, rank):
    features = ['passenger_count', 'trip_distance', 'RatecodeID',
                'PULocationID', 'DOLocationID', 'payment_type', 'extra', 'trip_duration']
    target = 'total_amount'

    X = df[features].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64).reshape(-1, 1)

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


def grads_to_vector(grads):
    parts = []
    shapes = {}
    for k in ('W1', 'b1', 'W2', 'b2'):
        arr = np.asarray(grads[k], dtype=np.float64)
        shapes[k] = arr.shape
        parts.append(arr.ravel())
    vec = np.concatenate(parts).astype(np.float64)
    return vec, shapes


def vector_to_grads(vec, shapes):
    grads = {}
    idx = 0
    for k in ('W1', 'b1', 'W2', 'b2'):
        size = np.prod(shapes[k])
        sub = vec[idx: idx + size]
        grads[k] = sub.reshape(shapes[k])
        idx += size
    return grads


def train_model(comm, model, X_train_local, y_train_local, batch_size_global=128, lr=1e-3,
                epochs=50, tol=1e-6, patience=5, seed=20):
    rank = comm.Get_rank()
    size = comm.Get_size()
    rng = np.random.RandomState(seed + rank * 997)

    n_train_local = X_train_local.shape[0]

    # batch数量均匀分布
    base = batch_size_global // size
    remainder = batch_size_global % size
    local_batch_target = base + (1 if rank < remainder else 0)
    if local_batch_target <= 0:
        local_batch_target = 1

    prev_loss = None
    no_improve = 0

    t0 = time.time() if rank == 0 else None

    # 全局训练样本数（各进程合并）
    total_train = comm.allreduce(n_train_local, op=MPI.SUM)

    for ep in range(epochs):
        local_batch_actual = min(local_batch_target, n_train_local)
        local_indices = rng.choice(n_train_local, size=local_batch_actual, replace=False)

        Xb = X_train_local[local_indices, :]
        yb = y_train_local[local_indices, :]
        y_pred_b, cache = model.forward_pass(Xb)
        local_grads = model.backward(y_pred_b, yb, cache)

        local_vec, shapes = grads_to_vector(local_grads)
        global_vec = np.zeros_like(local_vec)
        comm.Allreduce([local_vec, MPI.DOUBLE], [global_vec, MPI.DOUBLE], op=MPI.SUM)

        M_global = comm.allreduce(np.int64(local_batch_actual), op=MPI.SUM)

        grad_avg_vec = global_vec.astype(np.float64) / float(M_global)
        grad_avg = vector_to_grads(grad_avg_vec, shapes)

        model.apply_gradients(grad_avg, lr)

        y_pred_train_local, _ = model.forward_pass(X_train_local)
        diff_train_local = y_pred_train_local - y_train_local
        sse_local = np.sum(diff_train_local ** 2)

        sse_global = comm.allreduce(np.float64(sse_local), op=MPI.SUM)
        global_loss = 0.5 * sse_global / float(total_train)

        if rank == 0:
            print(
                f"Epoch {ep + 1}/{epochs} - global loss R = {global_loss:.6f} - M_global = {M_global} - time {time.time() - t0:.2f}s")

        if prev_loss is not None:
            if prev_loss - global_loss <= tol:
                no_improve += 1
            else:
                no_improve = 0
        prev_loss = global_loss
        if no_improve >= patience:
            if rank == 0:
                print(f"continuous {no_improve} epoch no improvements（tol={tol}），stop training。")
            break

    total_time = None
    if rank == 0:
        total_time = time.time() - t0

    return total_time


def compute_rmse_distributed(model, X_local, y_local, comm):
    n_local = X_local.shape[0]
    y_pred_local, _ = model.forward_pass(X_local)
    sse_local = np.sum((y_pred_local - y_local) ** 2)
    cnt_local = np.int64(n_local)
    sse_global = comm.allreduce(np.float64(sse_local), op=MPI.SUM)
    cnt_global = comm.allreduce(np.int64(cnt_local), op=MPI.SUM)
    rmse = np.sqrt(sse_global / float(cnt_global))
    return rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=32, help='number of hidden neurons')
    parser.add_argument('--activation', type=str, default='sigmoid', choices=['relu', 'tanh', 'sigmoid'],
                        help='activation function')
    parser.add_argument('--batch', type=int, default=128, help='global mini-batch size M')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training rounds')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # todo test on local machine, set 1000 to read files
    buff = split_file(comm, rank, 1000)
    data, input_dim = load_and_preprocess_data(buff, rank)
    X_train_local, X_test_local, y_train_local, y_test_local = global_fit_and_transform(data, comm, rank)
    model = NN(X_train_local.shape[1], args.hidden, args.activation, 42)

    train_time = train_model(comm, model, X_train_local, y_train_local,
                             batch_size_global=args.batch, lr=args.lr, epochs=args.epochs, tol=1e-6,
                             patience=10, seed=42)

    train_rmse = compute_rmse_distributed(model, X_train_local, y_train_local, comm)
    test_rmse = compute_rmse_distributed(model, X_test_local, y_test_local, comm)

    if rank == 0:
        print("training finished。")
        print(f"training time（rank 0）：{train_time:.2f}s")
        print(f"training RMSE: {train_rmse:.6f}")
        print(f"test RMSE: {test_rmse:.6f}")


if __name__ == "__main__":
    main()
