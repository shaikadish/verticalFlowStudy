import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer


class LSTM(nn.Module):
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

    Methods
    -------
    forward(x)
        Forward pass of data x through the model
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # Use GPU if available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
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

        try:
            out, _ = self.lstm(
                x.reshape(
                    x.size(0), x.size(1), x.size(2)), (h0, c0))
        except BaseException:
            out, _ = self.lstm(x.reshape(x.size(0), x.size(1), 1), (h0, c0))

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)

        out = self.fc(out)
        # out: (n, 10)
        return out
