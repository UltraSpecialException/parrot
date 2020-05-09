from torch_transformers import Transformer, TransformerLRScheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Optional, Union, Dict, Any, List
import json
import warnings


class ParrotConfig:
    """
    The configurations for the Parrot chatbot.
    """
    __required__  = {
        "vocab_size",
        "d_model",
        "d_ff",
        "num_heads",
        "num_layers",
        "used_embed_weights",
        "set_seq_len",
        "optimizer"
    }

    __type_check__ = {
        "vocab_size": int,
        "d_model": int,
        "d_ff": int,
        "num_heads": int,
        "num_layers": int,
        "dropout": float,
        "used_embed_weights": bool,
        "set_seq_len": int,
        "optimizer": str,
        "initial_lr": float,
        "weight_decay": float,
        "warmup_steps": int
    }

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 d_ff: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float=0.1,
                 use_embed_weights: bool=True,
                 set_seq_len: Optional[int]=None,
                 optimizer: Optional[str]=None,
                 initial_lr: Optional[float]=0.0,
                 weight_decay: Optional[float]=0.01,
                 warmup_steps: Optional[int]=4000,
                 **kwargs) -> None:
        """
        Initializes a Parrot configuration class.

        vocab_size: the number of unique words in our vocabulary
        d_model: the dimensionality of the outputs <hyperparameter>
        d_ff: the number of hidden units for our feed forward layer
            <hyperparameter>
        num_heads: the number of heads to split the attention down
            <hyperparameter>
        num_layers: the number of encoder layers to apply <hyperparameter>
        dropout: the rate of dropout regularization <hyperparameter>
        use_embed_weights: whether or not to use the embeddings' weights as the
            final projection weights
        set_seq_len: the longest sequence we ever have to process for
            pre-computation of the positional information
        optimizer: the optimizer (GD) algorithm to use
        optimizer_weights: the path to the checkpointed weights of the optimizer
        initial_lr: the initial learning rate of the optimizer
            <hyperparameter>
        weight_decay: the rate of decay of the magnitudes of the network's
            weights
        warmup_steps: the number of steps to warm up with to avoid early
            overfitting.
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_embed_weights = use_embed_weights
        self.set_seq_len = set_seq_len
        self.optimizer = optimizer

        if self.optimizer not in ["sgd", "adam"]:
            raise ValueError(f"Optimizer needs to be one of adam or sgd, "
                             f"{self.optimizer} was given")

        self.initial_lr = initial_lr if initial_lr is not None else 0.0
        self.weight_decay = weight_decay if weight_decay is not None else 0.01
        self.warmup_steps = warmup_steps if warmup_steps is not None else 4000

        self.specified_params = list(self.__type_check__.keys())

        if kwargs:
            print("Keyword arguments are given, the types of these arguments"
                  "cannot be checked, please ensure they are correct.")
            for keyword, value in kwargs.items():
                print(f"Received keyword {keyword} with value {value}")
                self.__dict__[keyword] = value
                self.specified_params.append(keyword)

        self.check_params()


    def check_params(self) -> None:
        """
        Check that the required parameters are not given as None.
        """
        required_params = set(self.__type_check__.keys())
        required_params.remove("set_seq_len")

        none_params = []
        for param in required_params:
            if self.__dict__[param] is None:
                none_params.append(param)

        if none_params:
            params_to_report = " ".join(none_params)
            raise ValueError(f"The following parameter(s) is/are required "
                             f"(not NoneType) but is/are detected to have "
                             f"value None: {params_to_report}")

        if self.optimizer == "Adam":
            self.check_optimizer_adam()

    def check_optimizer_adam(self) -> None:
        """
        If the optimizer chosen is Adam, check if the config has been given the
        correct associated parameters (betas, eps).
        """
        if "betas" not in self.__dict__:
            warnings.warn("Optimizer Adam is chosen but parameter betas is not "
                          "given, using default (0.9, 0.98)")
            self.__dict__["betas"] = (0.9, 0.98)

        if "eps" not in self.__dict__:
            warnings.warn("Optimizer Adam is chosen but parameter eps is not "
                          "given, using default 1e-9")
            self.__dict__["eps"] = 1e-9

    @classmethod
    def from_dict(cls, param_dict: Dict[str, Union[int, float, bool, str]]) \
            -> "ParrotConfig":
        """
        Return a Parrot config class from the given parameters dictionary.
        """
        expected_params = cls.__required__
        given_params = set(param_dict.keys())

        if expected_params - given_params:
            missing = []
            for param in expected_params:
                if param not in given_params:
                    missing.append(param)

            missing = " ".join(missing)
            raise ValueError(f"Given dictionary is missing the following "
                             f"parameters: {missing}")

        for param in param_dict:
            param_val = param_dict[param]
            if param in cls.__type_check__:
                if type(param_val) is cls.__type_check__["param"] or \
                        (param not in expected_params and param_val is None):
                    pass

                else:
                    given_type = type(param_val)
                    expected_type = cls.__type_check__["param"]
                    raise TypeError(f"Expected parameter {param} to have type"
                                    f"{expected_type} but was given type "
                                    f"{given_type}.")
            else:
                warnings.warn(f"Superfluous parameter {param} was given, hence"
                              f"skipped. Consider removing this parameter from"
                              f"the dictionary for subsequent runs.")

        return cls(**param_dict)


    def get_config_arguments(self) -> List[Any]:
        """
        Return a list of the configs as accepted as arguments into the
        transformer from torch_transformers. This method serves this sole
        purpose.

        Details found here: https://github.com/UltraSpecialException/
        torch-transformers/blob/master/modules/Transformer.py
        """
        return [
            self.vocab_size,
            self.d_model,
            self.d_ff,
            self.num_heads,
            self.num_layers,
            self.dropout,
            self.use_embed_weights,
            self.set_seq_len,
        ]

    def to_dict(self, to_json: bool=False, out_dir: Optional[str]=None) \
            -> Dict[str, Union[int, float, bool, str]]:
        """
        Export the current configurations into a dictionary and return it. If
        <to_json> is True, <out_dir> is required to be a valid path so that
        the dictionary of config can be exported into the file system.
        """
        config = {}
        for param in self.specified_params:
            config[param] = self.__dict__[param]

        if to_json:
            assert out_dir is not None, \
                "To save the configurations into a JSON, the parameter " \
                "out_dir needs to not be None."

            with open(out_dir) as config_file:
                json.dump(config, config_file)

        return config

    def get_optimizer(self, parameters) -> Union[optim.Adam, optim.SGD]:
        """
        Return an untrained optimizer given the configuration. <parameters> is
        the list of parameters that the optimizer need to step.
        """
        if self.optimizer == "Adam":
            # noinspection PyUnresolvedReferences
            return optim.Adam(
                parameters,
                lr=self.initial_lr,
                betas=self.betas,
                eps=self.eps
            )
        else:
            return optim.SGD(parameters, lr=self.initial_lr)


class Parrot:
    """
    A chatbot that is trained using the baseline Transformer architecture that
    uses data from a user's conversations to learn to mimic the user's talking
    and chatting styles.
    """

    def __init__(self,
                 config: Optional[ParrotConfig]=None,
                 vocab_size: Optional[int]=None,
                 d_model:  Optional[int]=None,
                 d_ff:  Optional[int]=None,
                 num_heads:  Optional[int]=None,
                 num_layers:  Optional[int]=None,
                 dropout: Optional[float]=None,
                 used_embed_weights: bool=True,
                 set_seq_len: Optional[int]=None,
                 optimizer: Optional[str]=None,
                 initial_lr: Optional[float]=None,
                 weight_decay: Optional[float]=None,
                 warmup_steps: Optional[int]=None,
                 device: torch.device=torch.device("cpu"),
                 **kwargs):
        """
        Initializes a Parrot chatbot.

        vocab_size: the number of unique words in our vocabulary
        d_model: the dimensionality of the outputs <hyperparameter>
        d_ff: the number of hidden units for our feed forward layer
            <hyperparameter>
        num_heads: the number of heads to split the attention down
            <hyperparameter>
        num_layers: the number of encoder layers to apply <hyperparameter>
        dropout: the rate of dropout regularization <hyperparameter>
        use_embed_weights: whether or not to use the embeddings' weights as the
            final projection weights
        set_seq_len: the longest sequence we ever have to process for
            pre-computation of the positional information
        optimizer: the optimizer (GD) algorithm to use
        initial_lr: the initial learning rate of the optimizer
        weight_decay: the rate of decay of the magnitudes of the network's
            weights
        warmup_steps: the number of steps to warm up with to avoid early
            overfitting.
        device: device which operations are performed on
        """
        if config is None:
            # if the config file is not given, we build the config file
            # from the given parameters, the init method of the config class
            # checks the parameters -- they need to all be given (all of them
            # need to not be None)
            config = ParrotConfig(vocab_size, d_model, d_ff, num_heads,
                                  num_layers, dropout, used_embed_weights,
                                  set_seq_len, optimizer, initial_lr,
                                  weight_decay, warmup_steps, **kwargs)

        self.config = config

        self.model = Transformer(*self.config.get_config_arguments(), device)

        model_params = list(self.model.named_parameters())
        non_decay = []
        decay = []

        for name, param in model_params:
            if "bias" in name:
                non_decay.append(param)
            else:
                decay.append(param)

        optim_params = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": non_decay}
        ]
        self.optimizer = self.config.get_optimizer(optim_params)
        self.lr_scheduler = TransformerLRScheduler(
            warmup_steps=config.warmup_steps,
            d_model=config.d_model,
            optimizer=self.optimizer
        )

        self.device = device

    def load_model_from_checkpoint(self, path: str) -> None:
        """
        From the given path, load the model checkpoint.
        """
        self.model.load_state_dict(torch.load(path, map_location="cpu"))

    def load_optimizer_from_checkpoint(self, path: str) -> None:
        """
        From the given path, load the optimizer checkpoint.
        """
        self.optimizer.load_state_dict(torch.load(path, map_location="cpu"))

    def load_scheduler_from_checkpint(self, path: str) -> None:
        """
        From the given path, load the LR scheduler checkpoint.
        """
        self.lr_scheduler.load_state_dict(torch.load(path, map_location="cpu"))

    def train(self, data_loader: DataLoader):
        """
        Train the Transformer model using data in the data loader.
        """
        pass
