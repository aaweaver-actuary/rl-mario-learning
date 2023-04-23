import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import random

from collections import deque

from MarioNet import MarioNet


class Mario:
    """
    Mario agent

    Parameters
    ----------
    state_dim : tuple
        dimensions of state, (C, H, W)
        Default is None, which will be set to (4, 84, 84)
    action_dim : int
        number of possible actions
        Example: we are allowing 2 actions: 0 move right, 1 jump right
        Default is 2
    save_dir : str
        directory to save model
        Default is "models"
    gamma : float
        Discount factor for future rewards. Must be a value between 0 and 1.
        This is used to calculate the discounted return, to ensure that the agent
        considers future rewards as well as current rewards, but helps make sure that
        the agent does not get stuck in a loop of trying to get a reward that is
        far away.
        Default is 0.9
    exploration_rate_decay : float
        Decay rate for exploration rate. Must be a value between 0 and 1.
        This is used to decay the exploration rate, so that the agent does not
        explore forever, but instead explores more at the beginning of training
        and less as it gets better at the game.
        Default is 0.99999975
    exploration_rate_min : float
        Minimum exploration rate. Must be a value between 0 and 1.
        This is used to set the minimum exploration rate, so that the agent does not
        stop exploring completely, but instead explores more at the beginning of training
        and less as it gets better at the game.
        Default is 0.1
    batch_size : int
        Number of samples to use for each training step.
        Default is 32
    save_each : int
        Number of steps to take before saving the model.
        Default is 5e5
    learning_rate : float
        Learning rate for optimizer. Must be a value between 0 and 1. Higher values
        will make the agent learn faster, but may cause it to miss the optimal
        solution. Lower values will make the agent learn slower, but will make it
        more likely to find the optimal solution.
        Default is 0.00025
    optimizer : torch.optim
        Optimizer to use for training. Must be a torch.optim object.
        Default is None, which will be set to torch.optim.Adam
    loss_fn : torch.nn
        Loss function to use for training. Must be a torch.nn object.
        Default is torch.nn.SmoothL1Loss(), which is a Huber loss function. This
        is a good loss function to use for Q-learning, because it is less sensitive
        to outliers than the MSE loss function.
    burnin : int
        Number of steps to take before starting to train the model. This is used to
        fill the replay memory with samples before starting to train the model.
        Default is 1e4
    learn_every : int
        Number of steps to take before training the model. This is used to train the
        model less frequently, so that the agent can explore more.
        Default is 3
    sync_every : int
        Number of steps to take before syncing the target network with the online
        network. This is used to update the target network less frequently, so that
        the agent can explore more.
        Default is 1e4
    memory_max_len : int
        Maximum number of samples to store in the replay memory. This is used to limit
        the amount of memory used by the replay memory.
        Default is 100000
    """
    def __init__(self
                , state_dim : tuple = None
                , action_dim : int = 2
                , save_dir : str = "models"
                , gamma : float = 0.9
                , exploration_rate_decay : float = 0.99999975
                , exploration_rate_min : float = 0.1
                , batch_size : int  = 32
                , save_each : int = 5e5
                , learning_rate : float = 0.00025
                , optimizer : torch.optim = None
                , loss_fn : torch.nn = torch.nn.SmoothL1Loss()
                , burnin : int = 1e4
                , learn_every : int = 3
                , sync_every : int = 1e4
                , memory_max_len : int = 100000
                ):
        # set parameters
        if state_dim is None:
            self.state_dim = (4, 84, 84)
        else:
            self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.gamma = gamma
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        self.batch_size = batch_size
        self.save_each = save_each
        self.learning_rate = learning_rate
        self.burnin = burnin
        self.learn_every = learn_every
        self.sync_every = sync_every
        self.memory_max_len = memory_max_len


        # initialize replay memory using deque, which is a double-ended queue
        # this is a list-like data structure that allows us to append and pop
        # from both ends of the list
        self.memory = deque(maxlen=self.memory_max_len)

        # set the device to cuda if available, otherwise cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()

        # set the network to the device
        self.net.to(device=self.device)

        # set optimizer 
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.net.parameters()
                , lr=self.learning_rate)
        else:
            self.optimizer = optimizer

        # set loss function
        self.loss_fn = loss_fn

        # at the beginning, Mario doesn't know what to do, so he explores
        # he learns as he goes, and eventually he stops exploring and starts
        # exploiting as he gets better at the game
        self.exploration_rate = 1

        # how much to decay the exploration rate after each step
        self.exploration_rate_decay = exploration_rate_decay
        
        # the minimum exploration rate -- Mario will always explore at
        # least this amount of the time
        self.exploration_rate_min = exploration_rate_min

        # the number of steps Mario has taken so far
        self.curr_step = 0

    def _update_exploration_rate(self):
        """
        Update the exploration rate, so that Mario explores less and less
        as he gets better at the game.
        """
        self.exploration_rate = self.exploration_rate * self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

    def _explore(self):
        """
        Explore the environment by taking a random action. This is used
        early in the training process, when Mario doesn't know what to do.
        """
        action_idx = np.random.randint(self.action_dim)
        return action_idx
    
    def _exploit(self, state):
        """
        Exploit the environment by taking the action that maximizes the
        Q-value. This is used later in the training process, when Mario
        has learned what to do.
        """
        # convert state to tensor
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).unsqueeze(0)

        # get Q-values from online network
        action_values = self.net(state, model="online")

        # choose action with highest Q-value
        return torch.argmax(action_values, axis=1).item()



    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action, meaning that 
        the action is chosen randomly with probability epsilon,
        and the action is chosen greedily with probability 1-epsilon.

        Greedily means that the action is chosen to maximize the
        Q-value.

        Randomly means that the action is chosen uniformly at
        random from the set of possible actions.

        Inputs:
            state(``LazyFrame``): the current state of the
                                  environment, dimension
                                  (self.state_dim)
        Outputs:
            action(``int``): the action to take, dimension (1)
        
        """
        # EXPLORE

        # test whether to explore or exploit
        if np.random.rand() < self.exploration_rate:
            action_idx = self._explore()
        else:
            action_idx = self._exploit(state)

        # regardless of whether you explore or exploit, decay the
        # exploration rate, since over time Mario should explore less
        # and exploit more
        self._update_exploration_rate()

        # increment step
        self.curr_step += 1
        return action_idx
    
    def _first_if_tuple(self, x):
        """
        Helper function to get the first element of a tuple.

        Used in the cache method.
        """
        out =  x[0] if isinstance(x, tuple) else x
        return out.__array__()

    def cache(self, state, next_state, action, reward, done):
        """
        Cache the experience in memory.

        Cache the experience in memory, which is a list of tuples
        of the form (state, action, reward, next_state, done).
        """       
        # state is a tuple of LazyFrames (take the first one)
        # and represents the current state of the environment
        # eg the image of the game before Mario takes an action
        # Mario learns by connecting the current state to the next
        # state based on the action he takes
        state = torch.tensor(self._first_if_tuple(state)
                             , device=self.device)

        # next_state is a tuple of LazyFrames (take the first one)
        # and represents the next state of the environment
        # eg the image of the game after Mario takes an action
        # Mario learns by connecting the current state to the next
        # state based on the action he takes
        next_state = torch.tensor(self._first_if_tuple(next_state)
                                  , device=self.device)

        # action is an integer representing the action Mario takes
        action = torch.tensor([action], device=self.device)

        # reward is a float representing the reward Mario gets for
        # taking the action
        # the reward is defined by the environment
        reward = torch.tensor([reward], device=self.device)

        # done is a boolean representing whether the episode is over
        # if done is True, then Mario has either won or lost the game
        done = torch.tensor([done], device=self.device)

        # add the experience to memory
        self.memory.append((state, next_state, action, reward, done,))


    def recall(self):
        """
        Retrieve a batch of experiences from memory and convert them
        to tensors
        """
        # each batch is a tuple of tensors that is randomly sampled
        # from memory
        batch = random.sample(self.memory, self.batch_size)

        # convert each batch element to a tensor
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        
        # return the batch of experiences to the agent
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def learn(self):
        """
        Learn from a batch of experiences. This is the main learning
        loop for the agent.

        The agent learns by sampling a batch of experiences from memory,
        and then using the batch to update the Q-network. The Q-network
        is updated by minimizing the loss between the TD estimate and
        the TD target.

        The TD estimate is the Q-value of the current state and action
        pair, and is calculated using the Q-network.
        """
        # update the target network every self.sync_every steps
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # save the model every self.save_each    steps
        if self.curr_step % self.save_each == 0:
            self.save()

        # don't learn until you've collected enough experiences
        if self.curr_step < self.burnin:
            return None, None

        # only learn every self.learn_every steps
        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    # there are two subvalues used to learn the Q-value:
    # the TD estimate and the TD target
    def td_estimate(self
                    , state: torch.Tensor = None
                    , action : torch.Tensor = None
                    ):
        """
        the TD estimate is the predicted optimal Q-value for a given
        state s and action a

        Parameters
        ----------
        state : torch.Tensor
            the current state of the environment, dimension
            (self.batch_size, self.state_dim)
        action : torch.Tensor
            the action taken in the current state, dimension
            (self.batch_size, 1)

        Returns
        -------
        current_Q : torch.Tensor
            the Q-value for the current state and action, dimension
            (self.batch_size, 1)
        """
        # get the current Q-value for the state and action
        # based on the online network, organized as a tensor,
        # where the first dimension is the batch size
        # and the second dimension is the number of actions
        # and returns the Q-value for the action taken, which is
        # what we are trying to predict
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]
        return current_Q


    # torch.no_grad() tells PyTorch not to track gradients because
    # we are not training the target network -- we are just using
    # it to calculate the TD target
    @torch.no_grad()
    def td_target(self
                  , reward : torch.Tensor = None
                  , next_state : torch.Tensor = None
                  , done : torch.Tensor = None
                  ):
        """
        the TD target is the optimal Q-value for the next state

        Parameters
        ----------
        reward : torch.Tensor (float)
            the reward for taking the action in the current state,
            dimension (self.batch_size, 1)
        next_state : torch.Tensor (float)
            the next state of the environment, dimension
            (self.batch_size, self.state_dim)
        done : torch.Tensor (bool)
            whether the episode is over, dimension (self.batch_size, 1)

        Returns
        -------
        td_target : torch.Tensor (float)
            the optimal Q-value for the next state, dimension
            (self.batch_size, 1)
        """
        print(f"Calculating TD target for {self.batch_size} experiences.")

        # get the Q-value for the next state based on the target
        next_state_Q = self.net(next_state, model="online")

        # get the action that maximizes the Q-value for the next state
        best_action = torch.argmax(next_state_Q, axis=1)

        # get the Q-value for the next state and best action
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]

        # calculate the TD target as the reward plus the discounted
        # Q-value for the next state (set to 0 if the episode is over)
        disccounted_next_Q = (1 - done.float()) * self.gamma * next_Q
        td_target = reward + disccounted_next_Q

        # return the TD target as a tensor (float)
        return (td_target).float()

    def update_Q_online(self
                        , td_estimate : torch.Tensor = None
                        , td_target : torch.Tensor = None
                        ):
        """
        Update the Q-network by minimizing the loss between the TD
        estimate and the TD target.

        Parameters
        ----------
        td_estimate : torch.Tensor (float)
            the predicted optimal Q-value for a given state s and
            action a, dimension (self.batch_size, 1)
        td_target : torch.Tensor (float)
            the optimal Q-value for the next state, dimension
            (self.batch_size, 1)

        Returns
        -------
        loss : float
            the loss between the TD estimate and the TD target
        """
        print(f"Updating Q-online at step {self.curr_step}")

        # calculate the loss between the TD estimate and the TD target
        loss = self.loss_fn(td_estimate, td_target)

        # backpropagate the loss through the Q-network
        self.optimizer.zero_grad()

        # calculate the gradients
        loss.backward()

        # update the weights
        self.optimizer.step()

        # return the loss as a float
        return loss.item()

    def sync_Q_target(self):
        """
        Update the target network by copying the weights from the
        online network. This is done every self.sync_every steps.
        """
        print(f"Syncing Q-target at step {self.curr_step}")
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        """
        Save the model to a checkpoint file. This is done every
        self.save_each steps.
        """
        # create the save directory if it doesn't exist
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_each)}.chkpt"
        )

        # save the model
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")
