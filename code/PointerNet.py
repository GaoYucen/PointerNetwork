import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder

        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim//2 if bidir else hidden_dim
        self.n_layers = n_layers*2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                hidden):
        """
        Encoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """

        embedded_inputs = embedded_inputs.permute(1, 0, 2)

        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units

        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # Initialize vector V

        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass

        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        alpha = self.softmax(att)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

        self.input_to_hidden_start = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden_start = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out_start = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att_start = Attention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask_start = Parameter(torch.ones(1), requires_grad=False)
        self.runner_start = Parameter(torch.zeros(1), requires_grad=False)

    def update_mask(self, original_mask, selected_end_pointers):
        """
        Update mask based on Rule 1.
        """
        batch_size, seq_len = original_mask.size()
        mask = original_mask.clone()
        for b in range(batch_size):
            # Rule 1: Update mask for already selected end pointers
            mask[b, selected_end_pointers[b]] = 0
        return mask

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        Decoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # Initialize chain for each sequence in the batch at the start of forward process
        chains = [{} for _ in range(batch_size)]
        # start_pointers = torch.arange(input_length).unsqueeze(0).repeat(batch_size, 1)

        # (batch, seq_len)
        mask = torch.ones(batch_size, input_length, dtype=torch.bool)
        self.att.init_inf(mask.size())

        mask_start = torch.ones(batch_size, input_length, dtype=torch.bool)
        self.att_start.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        runner_start = self.runner_start.repeat(input_length)
        for i in range(input_length):
            runner_start.data[i] = i
        runner_start = runner_start.unsqueeze(0).expand(batch_size, -1).long()

        end_outputs = []
        selected_end_pointers = [[] for _ in range(batch_size)] # Initialize as list of lists
        outputs_start = []
        pointers_start = []

        def get_dict_key(dic, value):
            keys = list(dic.keys())
            values = list(dic.values())
            idx = values.index(value)
            key = keys[idx]

            return key

        def find_chain_start(chain, start):
            """Traverse the chain from the given start until we find the start of the chain."""
            # visited = set()
            current = start
            while current in chain.values():
                # if current in visited: # Detected a cycle, should not happen
                #     return None
                # visited.add(current)
                current = get_dict_key(chain, current)
            return current

        def step(x, hidden):
            """
            Recurrence step function

            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * torch.tanh(c_t)

            # Attention section
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = torch.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        def step_start(x, hidden):
            """
            Recurrence step function

            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden_start(x) + self.hidden_to_hidden_start(h)
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * torch.tanh(c_t)

            # Attention section
            hidden_t, output = self.att_start(h_t, context, torch.eq(mask, 0))
            hidden_t = torch.tanh(self.hidden_out_start(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        decoder_input_start = decoder_input.clone()
        hidden_start = hidden

        # Recurrence loop
        for seq_idx in range(input_length):
            # print('mask_start', mask_start[0])
            # start
            h_t_start, c_t_start, outs_start = step_start(decoder_input_start, hidden_start)
            # print('outs_start', outs_start[0])
            hidden_start = (h_t_start, c_t_start)

            mask_start_tmp = mask_start.clone()
            for i in range(batch_size):
                mask_start_tmp[i] = torch.tensor([-1 if item == 0 else 1 for item in mask_start[i]])
            # print('mask_start_tmp', mask_start_tmp[0])
            masked_outs_start = outs_start * mask_start_tmp  # Convert mask to float to allow multiplication
            for i in range(batch_size):
                masked_outs_start[i] = torch.tensor([-1 if item == -1 else item1 for item, item1 in zip(mask_start_tmp[i], masked_outs_start[i])])
            # print('masked_outs_start', masked_outs_start[0])

            # Get maximum probabilities and indices
            max_probs_start, indices_start = masked_outs_start.max(1)
            # print('indices_start', indices_start[0])
            one_hot_pointers_start = (runner_start == indices_start.unsqueeze(1).expand(-1, outs_start.size()[1])).float()
            # print('one_hot_pointers_start', one_hot_pointers_start[0])

            # Update mask to ignore seen indices
            mask_start = mask_start * (1 - one_hot_pointers_start)


            # Get embedded inputs by max indices
            embedding_mask_start = one_hot_pointers_start.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            decoder_input_start = embedded_inputs[embedding_mask_start.data].view(batch_size, self.embedding_dim)

            outputs_start.append(outs_start.unsqueeze(0))
            pointers_start.append(indices_start.unsqueeze(1))

            temp_mask = mask.clone()  # Temporary mask for Rule 2 and Rule 3
            current_start_pointer = torch.tensor([item[0] for item in pointers_start[seq_idx].tolist()], dtype=int)
            # print('current_start_pointer', current_start_pointer)

            # Rule 2: Can't select the current start pointer
            temp_mask[torch.arange(batch_size), current_start_pointer] = 0
            
            # Rule 3: Can't select an endpoint forming a cycle
            for b in range(batch_size):
                if seq_idx != input_length - 1:  # if not the last sequence
                    chain_start = find_chain_start(chains[b], current_start_pointer[b].item())
                    temp_mask[b, chain_start] = 0

                # # Check if forming a cycle that is not the final cycle
                # if seq_idx != input_length - 1:  # if not the last sequence
                #     chain_end = chains[b][chain_start] if chain_start in chains[b] else None
                #     print('chain_end', chain_end)
                #     if chain_end == chain_start:  # If the chain has formed a cycle
                #         temp_mask[b, chain_end] = 0  # Block the end point that would form the cycle

            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * temp_mask.float() # Convert mask to float to allow multiplication

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            # Update chains and end_pointers
            for b in range(batch_size):
                chains[b][current_start_pointer[b].item()] = indices[b].item()
                selected_end_pointers[b].append(indices[b].item())
            
            # Update the original mask for Rule 1 at the end of the loop
            mask = self.update_mask(mask, selected_end_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).bool()
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)

            end_outputs.append(outs.unsqueeze(0))
            #end_pointers.append(indices.unsqueeze(1))

        end_outputs = torch.cat(end_outputs).permute(1, 0, 2)
        #end_pointers = torch.cat(end_pointers, 1)
        end_pointers = torch.tensor(selected_end_pointers, dtype=torch.long)
        start_outputs = torch.cat(outputs_start).permute(1, 0, 2)
        start_pointers = torch.cat(pointers_start, 1)

        return (start_outputs, start_pointers, end_outputs, end_pointers), hidden


class PointerNet(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False):
        """
        Initiate Pointer-Net

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.embedding = nn.Linear(2, embedding_dim)
        self.encoder = Encoder(embedding_dim,
                               hidden_dim,
                               lstm_layers,
                               dropout,
                               bidir)
        self.decoder = Decoder(embedding_dim, hidden_dim)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass

        :param Tensor inputs: Input sequence
        :return: Pointers probabilities and indices
        """

        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        inputs = inputs.view(batch_size * input_length, -1)
        embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs,
                                                       encoder_hidden0)
        if self.bidir:
            decoder_hidden0 = (torch.cat((encoder_hidden[0][0],encoder_hidden[0][1]), dim=-1),
                               torch.cat((encoder_hidden[0][0],encoder_hidden[0][1]), dim=-1))
        else:
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        (start_outputs, start_pointers, outputs, pointers), decoder_hidden = self.decoder(embedded_inputs,
                                                           decoder_input0,
                                                           decoder_hidden0,
                                                           encoder_outputs)

        return  start_outputs, start_pointers, outputs, pointers

        # outputs: probability distribution of each end point, [batch_size, sequence_length, num_classes], [100, 5, 5]
        # pointers: list of end points, [batch_size, sequence_length], [100, 5]