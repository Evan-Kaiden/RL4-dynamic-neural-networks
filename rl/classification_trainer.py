import torch
import torch.optim as optim
import torch.nn.functional as F

import random

class NetworkTrainer:
    def __init__(self, trainloader, testloader, agent, graph, exe, device='cpu'):
        self.opt = optim.Adam(graph.parameters(), lr=1e-3)

        self.device = device
        self.agent = agent
        self.trainloader = trainloader
        self.testloader = testloader
        self.exe = exe
        self.output_index = graph._outputs[0]

    @torch.no_grad()
    def act(self, obs):
        _, num_actions = obs

        return random.choice(range(num_actions))
        action_logits, _ = self.agent.forward(*obs)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        return action[0].item()
    
    def fit(self, epochs):
        for epoch in range(epochs):
            for image, targets in self.trainloader:
                image, targets = image.to(self.device), targets.to(self.device)

                obs, done = self.exe.reset_input(image)

                while not done:
                    obs, _, done = self.exe.step(self.act(obs))

                logits = self.exe.temps[self.output_index]

                loss = F.cross_entropy(logits, targets)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
            self.test(epoch)

    @torch.no_grad()  
    def test(self, epoch):
        total_loss = 0
        correct = 0
        total = 0
        for image, targets in self.testloader:
            image, targets = image.to(self.device), targets.to(self.device)

            obs, done = self.exe.reset_input(image)

            while not done:
                obs, _, done = self.exe.step(self.act(obs))

            logits = self.exe.temps[self.output_index]

            loss = F.cross_entropy(logits, targets)
            total_loss += loss

            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f"TEST | Epoch: {epoch} Loss: {total_loss/total:.3f}"
                  f"| Acc: {100.*correct/total:.3f}% ({correct}/{total})")



