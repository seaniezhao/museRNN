import DataPrepare
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#sequence_length = 64
input_size = 26
hidden_size = 64
num_layers = 2
num_classes = 2


learning_rate = 0.01


fileLoader = DataPrepare.FileLoader()



# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        # Decode the hidden state
        out = self.fc(out)
        return out

#low一点，暂时这么用
test_index = [267, 257, 526, 158, 141, 534, 250, 412, 508, 237, 180, 474, 155, 69, 436, 380, 21, 5, 369, 99, 318, 266, 485, 90, 317, 67, 157, 483, 419, 327, 146, 406, 214, 11, 347, 453, 113, 25, 285, 275, 464, 219, 332, 296, 259, 400, 352, 85, 7, 109, 159, 101, 547, 161, 131, 162, 248, 470, 173, 494, 112, 134, 524, 339, 529, 541, 230, 394, 12, 30, 396, 561, 232, 272, 565, 472, 264, 444, 126, 513, 186, 468, 142, 410, 242, 111, 149, 569, 51, 222, 39, 284, 566, 128, 200, 226, 171, 487, 403, 432]

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

model_path = 'model.ckpt'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
if model is None:

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    file_len = fileLoader.get_file_len()
    for i in range(file_len):
        if i in test_index:
            continue

        f = fileLoader.get_data_by_file(i)
        ft = torch.Tensor(f[1:,:])
        fl = torch.Tensor(f[0,:]).long()
        train_data = ft.reshape(-1, len(f[0]), input_size).to(device)
        labels = fl.to(device)

        # Forward pass
        outputs = model(train_data)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}]], Loss: {:.4f}'.format(i, file_len,  loss.item()))




# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for i in test_index:

        f = fileLoader.get_data_by_file(i)
        ft = torch.Tensor(f[1:, :])
        fl = torch.Tensor(f[0, :]).long()
        t_data = ft.reshape(-1, len(f[0]), input_size).to(device)
        labels = fl.to(device)

        outputs = model(t_data)

        for i in range(len(labels)):
            temp =outputs[i].numpy()
            predicted = np.argmax(temp, 0)
            if labels[i] == 1:
                total += 1
                if predicted == labels[i]:
                    correct += 1

    print(total)
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

