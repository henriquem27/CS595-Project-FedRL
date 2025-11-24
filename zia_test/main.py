import gymnasium as gym
import numpy as np

# simple policy network using numpy
class PolicyNetwork:
    def __init__(self, state_size, action_size):
        # initialize weights randomly
        self.weights = np.random.randn(state_size, action_size) * 0.01
        self.learning_rate = 0.01
        
    def predict(self, state):
        # compute action probabilities
        logits = np.dot(state, self.weights)
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def get_action(self, state):
        # sample action based on probabilities
        probs = self.predict(state)
        return np.random.choice(len(probs), p=probs)
    
    def update(self, states, actions, rewards):
        # compute discounted rewards
        discounted_rewards = self.discount_rewards(rewards)
        
        # normalize rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # update weights using policy gradient
        for state, action, reward in zip(states, actions, discounted_rewards):
            probs = self.predict(state)
            # gradient for policy gradient
            dsoftmax = probs.copy()
            dsoftmax[action] -= 1
            gradient = np.outer(state, dsoftmax)
            self.weights -= self.learning_rate * gradient * reward
    
    def discount_rewards(self, rewards, gamma=0.99):
        # calculate discounted cumulative rewards
        discounted = np.zeros_like(rewards)
        running_sum = 0
        for t in reversed(range(len(rewards))):
            running_sum = running_sum * gamma + rewards[t]
            discounted[t] = running_sum
        return discounted

# training function
def train(episodes=100, render_every=10):
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # create policy network
    policy = PolicyNetwork(state_size, action_size)
    
    print("starting training...")
    
    for episode in range(episodes):
        states, actions, rewards = [], [], []
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        # collect episode data
        while not (done or truncated):
            action = policy.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            episode_reward += reward
            
            state = next_state
        
        # update policy
        policy.update(states, actions, rewards)
        
        # print progress
        if episode % 10 == 0:
            print(f"episode {episode}: total reward = {episode_reward}")
    
    env.close()
    return policy

# test trained policy with visualization
def test_policy(policy, episodes=3):
    env = gym.make("CartPole-v1", render_mode="human")
    
    print("\ntesting trained policy with visualization...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action = policy.get_action(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            env.render()
        
        print(f"test episode {episode + 1}: total reward = {episode_reward}")
    
    env.close()

# run training and testing
if __name__ == "__main__":
    trained_policy = train(episodes=100)
    test_policy(trained_policy, episodes=3)
