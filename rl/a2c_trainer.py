import torch
import torch.optim as optim
import torch.nn.functional as F

from model.net_graph import NetGraph
from model.dag_executor import DAGExecutor
from rl.a2c_agent import A2CAgent

from tqdm import tqdm

from utils.viz import make_gif, make_path_graph

class A2CTrainer:
    def __init__(self, trainloader, 
                 testloader, 
                 agent : A2CAgent, 
                 graph : NetGraph, 
                 exe : DAGExecutor,
                 gamma : float = 0.99,
                 critic_coef : float = 0.5,
                 entropy_coef : float = 0.01,
                 device='cpu'):
        
        self.cls_opt = optim.Adam(graph.parameters(), lr=1e-3)
        self.rl_opt = optim.Adam(agent.parameters(), lr=1e-4)
        self.device = device
        self.agent = agent.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.exe = exe
        self.graph = graph
        self.output_index = graph._outputs[0]

        self.gamma = gamma
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef

    def act(self, obs):
        
        if isinstance(obs[0], torch.Tensor):
            obs = (obs[0].detach(), obs[1])

        action_logits, value = self.agent(*obs)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), value.squeeze(), log_prob.squeeze(), entropy.squeeze()
    
    def _discounted(self, rewards, gamma):
        R = torch.zeros_like(rewards[-1])
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.stack(returns)  # [T,B]
    
    def _rl_update(self, rewards, log_probs, values, entropies):
        # ----- Compute Returns -----
        returns = self._discounted(rewards, self.gamma).to(self.device).detach()
            
        # ----- Convert Lists to Tensors -----
        log_probs = torch.stack(log_probs).to(self.device)       
        values    = torch.stack(values).to(self.device)           
        entropies = torch.stack(entropies).to(self.device)
        
        advantages = (returns - values)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        
        
        # ----- Run policy update -----
        actor_loss  = -(log_probs * advantages).mean()
        critic_loss = 0.5 * (values - returns).pow(2).mean()
        entropy_term = entropies.mean()

        loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy_term

        self.rl_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.rl_opt.step()


    def _cls_update(self, logits, targets):
        loss = F.cross_entropy(logits, targets)
        self.cls_opt.zero_grad()
        loss.backward()
        self.cls_opt.step()

    def make_gifs(self, epoch, n):
        total = 0
        for image, _ in self.testloader:
            if image.size(0) > 1:
                for i in range(image.size(0)):
                    make_path_graph(self.agent, self.exe, self.graph, image[i, :, :, :].unsqueeze(0), filename=f'epoch{epoch}/dag_{total}.gif')
                    total += 1
                    if total >= n: break
            else:
                if total >= n: break
                total += 1
                make_path_graph(self.agent, self.exe, self.graph, image, filename=f'epoch{epoch}/dag_{total}.gif')
                
    def fit(self, epochs):
        for epoch in range(epochs):
            self.agent.train()
            pbar = tqdm(total=len(self.trainloader), desc=f'Epoch: {epoch}')
            total_reward = 0
            total = 0
            with pbar:
                for image, targets in self.trainloader:
                    image, targets = image.to(self.device), targets.to(self.device)
                    obs, done = self.exe.reset_input(image)
                    self.exe.reset_target(targets)
                    
                    log_probs, rewards, values, entropies = [], [], [], []

                    while not done:
                        action, value, log_prob, entropy = self.act(obs)
                        obs_, reward, done = self.exe.step(action)

                        log_probs.append(log_prob)
                        rewards.append(torch.as_tensor(reward, dtype=torch.float32, device=self.device))
                        values.append(value)
                        entropies.append(entropy)
                        
                        obs = obs_

                    logits = self.exe.temps[self.output_index]

                    self._rl_update(rewards, log_probs, values, entropies)
                    self._cls_update(logits, targets)

                    total_reward += sum(rewards)
                    total += 1
                    pbar.update(1)

            self.agent.eval()
            print(f"Epoch {epoch}, AVG Reward {total_reward / total:.4f}")
            self.make_gifs(epoch, 10)
            self.test(epoch)

    @torch.no_grad()  
    def test(self, epoch):
        total_loss = 0.0
        correct = 0
        total = 0
        for image, targets in self.testloader:
            image, targets = image.to(self.device), targets.to(self.device)

            obs, done = self.exe.reset_input(image)
            while not done:
                action, *_ = self.act(obs)
                obs, _, done = self.exe.step(action)
            
            logits = self.exe.temps[self.output_index]
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item() * targets.size(0)

            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f"TEST | Epoch: {epoch} Loss: {total_loss/total:.3f}"
              f"| Acc: {100.*correct/total:.3f}% ({correct}/{total})")