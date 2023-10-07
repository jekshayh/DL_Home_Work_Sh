import torch

#Star_System = torch.tensor([[distance_to_Stanton, Economy, Security, Exotic Goods]])
#Stanton = torch.tensor([[1.0, 1.0, 0.9, 0.4]])

Terra = torch.tensor([[0.9, 1.0, 1.0, 0.4]])
Magnus = torch.tensor([[0.9, 0.6, 0.5, 0.1]])
Kilian = torch.tensor([[0.7, 0.9, 1.0, 0.2]])
Pyro = torch.tensor([[0.9, 0.0, 0.0, 0.0]])
Goss = torch.tensor([[0.7, 0.8, 1.0, 0.3]])
Sol = torch.tensor([[0.5, 1.0, 1.0, 0.4]])
Gliese = torch.tensor([[0.2, 1.0, 0.7, 1.0]])

dataset = [
    (Terra, torch.tensor([[4.0]]), "Terra"),
    (Magnus, torch.tensor([[2.0]]), "Magnus"),
    (Kilian, torch.tensor([[3.0]]), "Kilian"),
    (Pyro, torch.tensor([[1.0]]), "Pyro"),
    (Goss, torch.tensor([[3.0]]), "Goss"),
    (Sol, torch.tensor([[4.0]]), "Sol"),
    (Gliese, torch.tensor([[5.0]]), "Gliese")
]

torch.manual_seed(2023)

weights = torch.rand((1, 4), requires_grad=True)
bias = torch.rand((1,1), requires_grad=True)

mse_loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD([weights, bias], lr=1e-5)

def predict_profit_score(obj: torch.Tensor) -> torch.Tensor:
    return obj @ weights.T + bias

def calc_loss(predicted_value: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    return mse_loss_fn(predicted_value, ground_truth)

num_epochs = 50
epoch_count = 0

for i in range(num_epochs):
    epoch_count += 1
    print(f"\nEpoch number {epoch_count}")
    for x, y, z in dataset:
        optimizer.zero_grad()
        threat_score = predict_profit_score(x)

        loss = calc_loss(threat_score, y)
        loss.backward()
        print(f"{z} - {loss}")
        optimizer.step()