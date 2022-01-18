import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer


class LSTM_regression(nn.Module):
    """
    LSTM model architecture

    Attributes
    ----------
    num_layers : int
        Sets the number of layers of the deep LSTM
    hidden_size : int
        Sets the size of the hidden state of the LSTM
    input_size : int
        Sets the number of input features to the LSTM
    num_classes : int
        Number of classes being classified
    device: str
        Sets whether using GPU or CPU
    num_flat: int
        Number of dense layers at output of LSTM, after MF and TSAT introduced

    Methods
    -------
    forward(x)
        Forward pass of data x through the model
    """

    def __init__(self, input_size, hidden_size,
                 num_layers, num_classes, num_flat):
        super(LSTM_regression, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_flat = num_flat
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        # Use GPU if available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc1_1 = nn.Linear(3, 1)

        self.fc2_1 = nn.Linear(3, hidden_size)
        self.fc2_2 = nn.Linear(hidden_size, 1)

        self.fc3_1 = nn.Linear(3, hidden_size)
        self.fc3_2 = nn.Linear(hidden_size, round(hidden_size / 2))
        self.fc3_3 = nn.Linear(round(hidden_size / 2), 1)

        self.fc4_1 = nn.Linear(3, hidden_size)
        self.fc4_2 = nn.Linear(hidden_size, round(hidden_size / 2))
        self.fc4_3 = nn.Linear(round(hidden_size / 2), round(hidden_size / 4))
        self.fc4_4 = nn.Linear(round(hidden_size / 4), 1)

    def forward(self, x, Tsat, mf):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size).to(
            self.device)
        c0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size).to(
            self.device)

        # x: (n, 28, 28), h0: (2, n, 128)
        try:
            # out, _ = self.lstm(x.reshape(x.size(0),x.size(2),x.size(1)),
            # (h0,c0)) #For single channel input
            out, _ = self.lstm(x.reshape(x.size(0), x.size(
                1), x.size(2)), (h0, c0))  # for 7 channel input
        except BaseException:
            out, _ = self.lstm(
                x.reshape(
                    x.size(0), x.size(1), x.size(2)), (h0, c0))

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)

        out = self.fc(out)
        # out: (n, 10)
        out = torch.stack((out.squeeze(), Tsat.squeeze(), mf.squeeze()))
        out = torch.transpose(out, 0, 1)

        if(self.num_flat == 1):
            out = self.fc1_1(out)
        elif(self.num_flat == 2):
            out = self.fc2_1(out)
            out = self.fc2_2(out)
        elif(self.num_flat == 3):
            out = self.fc3_1(out)
            out = self.fc3_2(out)
            out = self.fc3_3(out)
        elif(self.num_flat == 4):
            out = self.fc4_1(out)
            out = self.fc4_2(out)
            out = self.fc4_3(out)
            out = self.fc4_4(out)

        return out
