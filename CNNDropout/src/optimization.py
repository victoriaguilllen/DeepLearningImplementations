# deep learning libraries
import torch

# other libraries
from typing import Iterator, Dict, Any, DefaultDict


class SGD(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self, params: Iterator[torch.nn.Parameter], lr=1e-3, weight_decay: float = 0.0
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay)

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO
        # recorro cada grupo de parámetros (cada vector wi pertenece a un grupo)
        for group in self.param_groups:
            # accedo a los parámetros del grupo
            for param in group["params"]:
                # gradientes
                param_grad = param.grad
                if param_grad is None:
                    continue

                # miro si hay weight decay
                if group["weight_decay"] != 0:
                    param_grad.data.add_(param.data, alpha=group["weight_decay"])

                # actualizo los pesos
                param.data.add_(param_grad.data, alpha=-group["lr"])


class SGDMomentum(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        # define defaults
        self.defaults: Dict[Any, Any] = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum
        )

        # call super class constructor
        super().__init__(params, self.defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Attr:
            param_groups: list with the dict of the parameters.
            state: dict with the state for each parameter.
        """

        # TODO
        for group in self.param_groups:
            for param in group[
                "params"
            ]:  # cada elemento de 'params' está asociando a un elemento del diccionario 'state'
                # gradiente del parámetro
                param_grad = param.grad

                # comprobar si tiene gradiente
                if param_grad is None:
                    continue

                # comprobar si se le aplica el weight decay
                if group["weight_decay"] != 0:
                    param_grad.data.add_(
                        param.data, alpha=self.defaults["weight_decay"]
                    )

                # comprobar si se le aplica el momentum
                if "momentum_buffer" not in self.state[param]:
                    buf = self.state[param]["momentum_buffer"] = torch.clone(
                        param_grad.data
                    )  # .detach ¿Haría faltaaaaaaaa?
                else:
                    buf = self.state[param]["momentum_buffer"]
                    buf.mul_(self.defaults["momentum"]).add_(param_grad.data)

                # actualizamos pesos
                param.data.add_(buf, alpha=-group["lr"])


class SGDNesterov(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        # define defaults
        self.defaults: Dict[Any, Any] = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum
        )

        # call super class constructor
        super().__init__(params, self.defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO
        for group in self.param_groups:
            for param in group["params"]:
                # obtengo el gradiente del param
                param_grad = param.grad

                # comprobar si tiene gradiente
                if param_grad is None:
                    continue

                # miro si necesita weight decay
                if group["weight_decay"] != 0:
                    param_grad.data.add_(
                        param.data, alpha=self.defaults["weight_decay"]
                    )

                # miro si necesita momentum
                if "momentum_buffer" not in self.state[param]:
                    buf = self.state[param]["momentum_buffer"] = torch.clone(
                        param_grad.data
                    ).detach()  # .detach ¿Haría faltaaaaaaaa?
                else:
                    buf = self.state[param]["momentum_buffer"]
                    buf.mul_(self.defaults["momentum"]).add_(param_grad.data)

                nesterov = param_grad.data.add_(buf, alpha=group["momentum"])
                param.data.add_(nesterov, alpha=-group["lr"])


class Adam(torch.optim.Optimizer):
    """
    This class is a custom implementation of the Adam algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # TODO
        self.defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, self.defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]

                # inicializamos
                if len(state) == 0:
                    state["step"] = 0
                    # primer momento
                    state["mt"] = torch.zeros_like(param.data)
                    # segundo momento
                    state["vt"] = torch.zeros_like(param.data)

                mt, vt = state["mt"], state["vt"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # miramos si necesita weight decay
                if group["weight_decay"] != 0:
                    grad.add_(param.data, alpha=group["weight_decay"])

                mt.mul_(beta1).add_(grad, alpha=1 - beta1)
                vt.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = vt.sqrt().add_(group["eps"])
                step_size = (
                    group["lr"]
                    * (1 - beta2 ** state["step"]) ** 0.5
                    / (1 - beta1 ** state["step"])
                )

                param.data.addcdiv_(mt, denom, value=-step_size)
