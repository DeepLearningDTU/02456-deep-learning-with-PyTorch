import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_generator import generate

# TODO we're not using targets masks from the data set. We train to predict zeros after the EOS character


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)

NUM_INPUTS = 27
NUM_OUTPUTS = 11  # (0-9 + '#')

### Hyperparameters and general configs
MAX_SEQ_LEN = 8
MIN_SEQ_LEN = 5
BATCH_SIZE = 16
TRAINING_SIZE = 80000
LEARNING_RATE = 0.003
# Hidden size of enc and dec need to be equal if last hidden of encoder becomes init hidden of decoder
# Otherwise we would need e.g. a linear layer to map to a space with the correct dimension
NUM_UNITS_ENC = NUM_UNITS_DEC = 48
TEST_SIZE = 2000
EPOCHS = 10
MODEL_PATH = "saved_model/encoder_decoder_model"

RNN_TYPE = 'lstm'  # 'gru'
TEACHER_FORCING = False

assert TRAINING_SIZE % BATCH_SIZE == 0


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, self.hidden_size)
        if RNN_TYPE == 'gru':
            rnn = nn.GRU
        elif RNN_TYPE == 'lstm':
            rnn = nn.LSTM
        self.rnn = rnn(self.hidden_size, self.hidden_size, batch_first=True)

    def forward(self, inputs, hidden):
        # Input shape [batch, seq_in_len]
        inputs = inputs.long()
        # Embedded shape [batch, seq_in_len, embed]
        embedded = self.embedding(inputs)
        # Output shape [batch, seq_in_len, embed]
        # Hidden shape [1, batch, embed], last hidden state of the GRU cell
        # We will feed this last hidden state into the decoder
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        init = torch.zeros(1, batch_size, self.hidden_size, device=device)
        if RNN_TYPE == 'gru':
            return init
        if RNN_TYPE == 'lstm':
            return init, init


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        if RNN_TYPE == 'gru':
            rnn = nn.GRU
        elif RNN_TYPE == 'lstm':
            rnn = nn.LSTM
        self.rnn = rnn(self.hidden_size, self.hidden_size, batch_first=True)

    def forward(self, inputs, hidden, output_len, teacher_forcing=False):
        # Input shape: [batch, output_len]
        # Hidden shape: [seq_len=1, batch_size, hidden_dim] (the last hidden state of the encoder)

        if teacher_forcing:
            dec_input = inputs
            embed = self.embedding(dec_input)   # shape [batch, output_len, hidden_dim]
            out, hidden = self.rnn(embed, hidden)
            out = self.out(out)  # linear layer, out has now shape [batch, output_len, output_size]
            output = F.log_softmax(out, -1)
        else:
            # Take the EOS character only, for the whole batch, and unsqueeze so shape is [batch, 1]
            # This is the first input, then we will use as input the GRU output at the previous time step
            dec_input = inputs[:, 0].unsqueeze(1)

            output = []
            for i in range(output_len):
                out, hidden = self.rnn(self.embedding(dec_input), hidden)
                out = self.out(out)  # linear layer, out has now shape [batch, 1, output_size]
                out = F.log_softmax(out, -1)
                output.append(out.squeeze(1))
                out_symbol = torch.argmax(out, dim=2)   # shape [batch, 1]
                dec_input = out_symbol   # feed the decoded symbol back into the recurrent unit at next step

            output = torch.stack(output).permute(1, 0, 2)  # [batch_size x seq_len x output_size]

        return output


def forward_pass(encoder, decoder, x, t, t_in, criterion, max_t_len, teacher_forcing):
    """
    Executes a forward pass through the whole model.

    :param encoder:
    :param decoder:
    :param x: input to the encoder, shape [batch, seq_in_len]
    :param t: target output predictions for decoder, shape [batch, seq_t_len]
    :param criterion: loss function
    :param max_t_len: maximum target length

    :return: output (after log-softmax), loss, accuracy (per-symbol)
    """
    # Run encoder and get last hidden state (and output)
    batch_size = x.size(0)
    enc_h = encoder.init_hidden(batch_size)
    enc_out, enc_h = encoder(x, enc_h)

    dec_h = enc_h  # Init hidden state of decoder as hidden state of encoder
    dec_input = t_in
    out = decoder(dec_input, dec_h, max_t_len, teacher_forcing)
    out = out.permute(0, 2, 1)
    # Shape: [batch_size x num_classes x out_sequence_len], with second dim containing log probabilities

    loss = criterion(out, t)
    pred = get_pred(log_probs=out)
    accuracy = (pred == t).type(torch.FloatTensor).mean()
    return out, loss, accuracy


def train(encoder, decoder, inputs, targets, targets_in, criterion, enc_optimizer, dec_optimizer, epoch, max_t_len):
    encoder.train()
    decoder.train()
    for batch_idx, (x, t, t_in) in enumerate(zip(inputs, targets, targets_in)):
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        x = x.to(device)
        t = t.to(device).long()
        t_in = t_in.to(device).long()

        _, loss, accuracy = forward_pass(encoder, decoder, x, t, t_in, criterion, max_t_len,
                                         teacher_forcing=TEACHER_FORCING)

        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()

        if batch_idx % 200 == 0:
            print('Epoch {} [{}/{} ({:.0f}%)]\tTraining loss: {:.4f} \tTraining accuracy: {:.1f}%'.format(
                epoch, batch_idx * len(x), TRAINING_SIZE,
                100. * batch_idx * len(x) / TRAINING_SIZE, loss.item(),
                100. * accuracy.item()))

        loss = torch.autograd.Variable(loss, requires_grad=True)
        loss.backward()




def test(encoder, decoder, inputs, targets, targets_in, criterion, max_t_len):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        targets = targets.long().to(device)
        targets_in = targets_in.long().to(device)
        out, loss, accuracy = forward_pass(encoder, decoder, inputs, targets, targets_in, criterion, max_t_len,
                                           teacher_forcing=TEACHER_FORCING)
    return out, loss, accuracy


def main():

    encoder = EncoderRNN(NUM_INPUTS, NUM_UNITS_ENC).to(device)
    decoder = DecoderRNN(NUM_UNITS_DEC, NUM_OUTPUTS).to(device)
    enc_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    # Get training set
    inputs, _, targets_in, targets, targets_seqlen, _, _, _, text_targ = generate(TRAINING_SIZE, min_len=MIN_SEQ_LEN, max_len=MAX_SEQ_LEN)
    max_target_len = max(targets_seqlen)
    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)
    targets_in = torch.tensor(targets_in)
    unique_text_targets = set(text_targ)

    # Get validation set
    val_inputs, _, val_targets_in, val_targets, val_targets_seqlen, _, val_text_in, _, val_text_targ = \
        generate(TEST_SIZE, min_len=MIN_SEQ_LEN, max_len=MAX_SEQ_LEN, invalid_set=unique_text_targets)
    val_inputs = torch.tensor(val_inputs)
    val_targets = torch.tensor(val_targets)
    val_targets_in = torch.tensor(val_targets_in)
    max_val_target_len = max(val_targets_seqlen)
    test(encoder, decoder, val_inputs, val_targets, val_targets_in, criterion, max_val_target_len)


    # Split training set in batches
    inputs = [inputs[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]
    targets = [targets[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]
    targets_in = [targets_in[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]

    # Quick and dirty - just loop over training set without reshuffling
    for epoch in range(1, EPOCHS + 1):
        train(encoder, decoder, inputs, targets, targets_in, criterion, enc_optimizer, dec_optimizer, epoch, max_target_len)
        _, loss, accuracy = test(encoder, decoder, val_inputs, val_targets, val_targets_in, criterion, max_val_target_len)
        print('\nTest set: Average loss: {:.4f} \tAccuracy: {:.3f}%\n'.format(loss, accuracy.item()*100.))

        # Show examples
        print("Examples: prediction | input")
        out, _, _ = test(encoder, decoder, val_inputs[:10], val_targets[:10], val_targets_in[:10], criterion, max_val_target_len)
        pred = get_pred(out)
        pred_text = [numbers_to_text(sample) for sample in pred]
        for i in range(10):
            print(pred_text[i], "\t", val_text_in[i])
        print()

    # torch.save(model.state_dict(), MODEL_PATH)


def numbers_to_text(seq):
    return "".join([str(to_np(i)) if to_np(i) != 10 else '#' for i in seq])

def to_np(x):
    return x.cpu().numpy()

def get_pred(log_probs):
    """
    Get class prediction (digit prediction) from the net's output (the log_probs)
    :param log_probs: Tensor of shape [batch_size x n_classes x sequence_len]
    :return:
    """
    return torch.argmax(log_probs, dim=1)


if __name__ == '__main__':
    main()
