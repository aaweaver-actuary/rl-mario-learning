import torch
import torch.nn as nn
import copy


class MarioNet(nn.Module):
    """
    mini CNN structure
    input -> 
        (conv2d + relu) x 3 -> 
        flatten -> 
        (dense + relu) x 2 -> 
        output
  """

    def __init__(self
                 , input_dim : tuple
                 , output_dim : int
                 , expected_height : int = 84
                 , expected_width : int = 84
                 , kernel_size : list = None
                 , stride : list = None
                 , linear_nodes : int = 512
                 , relu_slope : float = 0.1
                 , dropout : float = 0.2
                 , ):
        super().__init__()
        # input_dim: (C, H, W)
        # C: number of channels
        # H: height
        # W: width
        c, h, w = input_dim

        # check that input dimensions are correct (84x84x4)
        # 4 channels because we are using 4 frames as input (FrameStack)
        if h != expected_height:
            raise ValueError(f"Expecting input height: {expected_height}, got: {h}")
        if w != expected_width:
            raise ValueError(f"Expecting input width: {expected_width}, got: {w}")
        
        # default kernel sizes
        if kernel_size is None:
            kernel_size = [8, 4, 3]

        # default strides
        if stride is None:
            stride = [4, 2, 1]

        # define network
        self.online = nn.Sequential(
            
            # 3 convolutional layers w/ leaky relu activation
            nn.Conv2d(in_channels=c
                    , out_channels=32
                    , kernel_size=kernel_size[0]
                    , stride=stride[0]),
            nn.LeakyReLU(relu_slope),

            nn.Conv2d(in_channels=32
                    , out_channels=64
                    , kernel_size=kernel_size[1]
                    , stride=stride[1]),
            nn.LeakyReLU(relu_slope),

            nn.Conv2d(in_channels=64
                    , out_channels=64
                    , kernel_size=kernel_size[2]
                    , stride=stride[2]),
            nn.LeakyReLU(relu_slope),

            # flatten output
            nn.Flatten(),

            # 2 dense layers w/ leaky relu activation and dropout
            nn.Linear(3136, linear_nodes),
            nn.LeakyReLU(relu_slope),
            nn.Dropout(dropout),
            nn.Linear(linear_nodes, output_dim),
        )

        # initialize target network
        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen, so that they are not updated by the optimizer
        for p in self.target.parameters():
            # requires grad is a boolean flag that indicates whether
            # the parameters should be updated by the optimizer or not
            p.requires_grad = False

    # forward pass through network
    def forward(self
                , input : torch.Tensor
                , model : str = "online"
                ):
        # print(f"Forward pass through {model} network")
        # print(f"Input shape: {input.shape}")
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)