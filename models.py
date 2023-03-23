from typing import Any, List, Optional, Tuple, Union

import torch
from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP, ModelList, ModelListGP, SingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.input import (
    AffineInputTransform,
    ChainedInputTransform,
    Log10,
    Normalize,
)
from botorch.models.transforms.outcome import Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import LinearKernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.kernels.polynomial_kernel import PolynomialKernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.models import ExactGP
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor
from utils import SustainableConcreteDataset, get_bounds, get_day_zero_data


class SustainableConcreteModel(object):
    def __init__(
        self,
        strength_model: Optional[Model] = None,
        gwp_model: Optional[Model] = None,
        d: Optional[int] = None,
    ):
        self.strength_model = None
        self.gwp_model = None
        self.d = d

    def fit_strength_model(self, data: SustainableConcreteDataset) -> None:
        X, Y, Yvar, X_bounds = data.strength_data
        self._set_d(X.shape[-1])
        self.strength_model = fit_strength_gp(X, Y, Yvar, X_bounds)

    def fit_gwp_model(self, data: SustainableConcreteDataset) -> None:
        X, Y, Yvar, X_bounds = data.gwp_data
        self._set_d(X.shape[-1] + 1)
        self.gwp_model = fit_gwp_gp(X, Y, Yvar, X_bounds)

    def _set_d(self, d: int) -> None:
        if self.d is None:
            self.d = d

    def get_model_list(self) -> ModelListGP:
        """Returns a ModelListGP modeling the GWP and compressive strength objectives as a function
        of composition only.
        Converts the strength and gwp models into a model list of independent models for gwp,
        and x-day strengths, by fixing the time input of the strength model at 1 and 28 days.
        """
        models = [
            self.gwp_model,
            FixedFeatureModel(
                base_model=self.strength_model, dim=self.d, indices=[self.d - 1], values=[1]
            ),  # strength at day 1
            FixedFeatureModel(
                base_model=self.strength_model, dim=self.d, indices=[self.d - 1], values=[28]
            ),  # strength at day 28
        ]
        model = ModelList(*models)
        return model  # for use with multi-objective optimization

    def plot_strength_curve(self, composition: Tensor, max_day: int = 28) -> None:
        time = torch.arange(max_day + 1)
        composition = composition.unsqueeze(0).expand(len(time))
        # IDEA: use FixedFeatureModel?


# BatchedMultiOutputGPyTorchModel, ExactGP
class FixedFeatureModel(Model):
    # advantage: only need to implement posterior for it to work with qNEHI
    # disadvantage: makes the strength outputs independent
    # TODO: check that these are appended before the InputTransforms are applied, not after.
    def __init__(
        self,
        base_model: Model,
        dim: int,
        indices: Union[List[int], Tensor],
        values: Union[List[float], Tensor],
    ):
        super().__init__()
        self.base_model = base_model
        if len(indices) != len(values):
            raise ValueError("indices and values do not have the same length.")
        indices = torch.as_tensor(indices)
        values = torch.as_tensor(values)
        self._indices = indices
        self._fixed = torch.tensor([i in indices for i in torch.arange(dim, dtype=indices.dtype)])
        self._values = values

    def _add_fixed_features(self, X: Tensor):
        """
        Input:
            - X: A (n x d)-dim Tensor.
        Output:
            A (n x (d + len(self._indices)))-dim Tensor.
        """
        Z = torch.zeros(*X.shape[:-1], X.shape[-1] + len(self._indices))
        Z[..., self._fixed] = self._values
        Z[..., ~self._fixed] = X
        return Z

    def forward(self, X: Tensor, *args, **kwargs):
        return self.base_model.forward(self._add_fixed_features(X), *args, **kwargs)

    def posterior(self, X: Tensor, *args, **kwargs):
        return self.base_model.posterior(self._add_fixed_features(X), *args, **kwargs)


def fit_gwp_gp(X: Tensor, Y: Tensor, Yvar: Tensor, X_bounds: Tensor):
    """
    Input:
        X: Tensor of composition inputs without time (n x d).
        Y: Tensor of GWP values (n x 1).
        Yvar: Tensor of GWP variances (n x 1).

    Output:
        A FixedNoiseGP model fit to the data.
    """
    d_in = X.shape[-1]
    d_out = Y.shape[-1]
    if d_out != 1:
        raise ValueError("Output dimensions is not one in gwp fitting.")
    # GWP is a linear function of the inputs
    kernel = LinearKernel()
    model = FixedNoiseGP(
        train_X=X,
        train_Y=Y,
        train_Yvar=Yvar,
        covar_module=kernel,
        input_transform=Normalize(d_in, bounds=X_bounds),
        outcome_transform=Standardize(d_out),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def fit_strength_gp(X: Tensor, Y: Tensor, Yvar: Tensor, X_bounds: Tensor):
    """
    Input:
        X: Tensor of composition inputs including time (n x d).
        Y: Tensor of strength values (n x 1).
        Yvar: Tensor of strength variances (n x 1).

    Output:
        A FixedNoiseGP model fit to the data.
    """
    d_in = X.shape[-1]
    d_out = Y.shape[-1]
    if d_out != 1:
        raise ValueError("Output dimensions is not one in strength curve fitting.")

    # add data to condition GP to be zero at day zero
    X_0, Y_0, Yvar_0 = get_day_zero_data(bounds=X_bounds, n=128)
    X = torch.cat((X, X_0), dim=0)
    Y = torch.cat((Y, Y_0), dim=0)
    Yvar = torch.cat((Yvar, Yvar_0), dim=0)

    # joint kernel to model all interactions
    base_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=d_in,
        lengthscale_constraint=None,
        lengthscale_prior=None,
    )
    scaled_base_kernel = ScaleKernel(
        base_kernel=base_kernel,
        outputscale_prior=GammaPrior(2.0, 0.15),
    )

    # additive kernel to model behavior w.r.t. time
    # try matern?
    time_kernel = RBFKernel(  # MaternKernel # smoother RBF seems to work better for additive components
        active_dims=torch.tensor([d_in - 1]),  # last dimension is time
        ard_num_dims=1,
        lengthscale_constraint=None,
        # lengthscale_prior=None,
    )
    scaled_time_kernel = ScaleKernel(
        base_kernel=time_kernel,
        outputscale_prior=GammaPrior(2.0, 0.15),
        # batch_shape=batch_shape,
    )

    # IDEA: + scaled_water_kernel and other additive components
    kernel = scaled_base_kernel + scaled_time_kernel
    # model = FixedNoiseGP(
    #     train_X=X,
    #     train_Y=Y,
    #     train_Yvar=Yvar,
    #     covar_module=kernel,
    #     input_transform=get_strength_gp_input_transform(X_bounds),
    #     outcome_transform=Standardize(d_out),
    # )
    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=kernel,
        input_transform=get_strength_gp_input_transform(X_bounds),
        outcome_transform=Standardize(d_out),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def get_strength_gp_input_transform(bounds: Tensor):
    """Chains a log(time + 1) and Normalize transform on d dimensional input data,
    with the provided bounds.
    """
    d = bounds.shape[-1]
    time_index = [d - 1]
    tf1 = AffineInputTransform(  # adds one to time dimension before taking log
        d,
        coefficient=torch.ones(1),
        offset=torch.ones(1),
        indices=time_index,
        reverse=True,
    )
    tf2 = Log10(indices=time_index)  # taking log of time dimension for better extrapolation
    transformed_bounds = tf2(tf1(bounds))
    tf3 = Normalize(d, bounds=transformed_bounds)  # normalizing after log(t + 1) transform
    return ChainedInputTransform(tf1=tf1, tf2=tf2, tf3=tf3)
