import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_inout_sequences(input_data, tw, output_window=1):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i : i + tw]
        train_label = input_data[i + output_window : i + tw + output_window]
        inout_seq.append((train_seq, train_label))

    return torch.FloatTensor(inout_seq)


def split_data(data, split=0.6, input_window=10, output_window=1):
    """
    * input_window: number of input steps
    * output_window: number of prediction steps, in this model its fixed to one
    """

    series = data

    split = round(split * len(series))

    train_data = series[:split]
    test_data = series[split:]

    train_data = train_data.cumsum()
    test_data = test_data.cumsum()

    # Training data augmentation, increase amplitude for the model to better generalize.(Scaling by 2 is aribitrary)
    # Similar to image transformation to allow model to train on wider data sets
    train_data = 2 * train_data

    train_sequence = create_inout_sequences(train_data, input_window, output_window=output_window)
    train_sequence = train_sequence[:-output_window]

    test_data = create_inout_sequences(test_data, input_window, output_window=output_window)
    test_data = test_data[:-output_window]

    return train_sequence.to(DEVICE), test_data.to(DEVICE)