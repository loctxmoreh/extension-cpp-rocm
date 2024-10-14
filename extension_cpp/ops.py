from typing import Tuple
import torch
from torch import Tensor

__all__ = ["lltm", "reference_lltm", "mymuladd", "myadd_out"]


def lltm(
    input: Tensor, weights: Tensor, bias: Tensor, old_h: Tensor, old_cell: Tensor
) -> Tuple[Tensor, Tensor]:
    return LLTMFunction.apply(input, weights, bias, old_h, old_cell)


class LLTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = torch.ops.extension_cpp.lltm_forward.default(
            input, weights, bias, old_h, old_cell
        )
        new_h, new_cell = outputs[:2]
        variables = list(outputs[1:]) + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_h, grad_cell):
        (
            d_old_h,
            d_input,
            d_weights,
            d_bias,
            d_old_cell,
        ) = torch.ops.extension_cpp.lltm_backward.default(
            grad_h, grad_cell, *ctx.saved_tensors
        )
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


@torch.library.impl_abstract("extension_cpp::lltm_forward")
def _(input, weights, bias, old_h, old_cell):
    X = torch.cat([old_h, input], dim=1)
    gate_weights = torch.nn.functional.linear(X, weights, bias)
    gates = gate_weights.chunk(3, dim=1)
    input_gate = torch.empty_like(gates[0])
    output_gate = torch.empty_like(gates[1])
    candidate_cell = torch.empty_like(gates[2])
    new_cell = torch.empty_like(old_cell)
    new_h = torch.empty_like(old_h)
    if input.device.type == "cuda":
        batch_size = old_cell.shape[0]
        state_size = old_cell.shape[1]
        gate_weights = gate_weights.reshape(batch_size, 3, state_size)
    return new_h, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights


def reference_lltm(
    input: Tensor, weights: Tensor, bias: Tensor, old_h: Tensor, old_cell: Tensor
) -> Tuple[Tensor, Tensor]:
    X = torch.cat([old_h, input], dim=1)

    # Compute the input, output and candidate cell gates with one MM.
    gate_weights = torch.nn.functional.linear(X, weights, bias)
    # Split the combined gate weight matrix into its components.
    gates = gate_weights.chunk(3, dim=1)

    input_gate = torch.sigmoid(gates[0])
    output_gate = torch.sigmoid(gates[1])
    # Here we use an ELU instead of the usual tanh.
    candidate_cell = torch.nn.functional.elu(gates[2])

    # Compute the new cell state.
    new_cell = old_cell + candidate_cell * input_gate
    # Compute the new hidden state and output.
    new_h = torch.tanh(new_cell) * output_gate

    return new_h, new_cell


# (Additional) mymul and mymulladd

class MyMulAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(a, b, c):
        return torch.ops.extension_cpp.mymuladd.default(a, b, c)

    @staticmethod
    def setup_context(ctx, inputs, output):
        a, b, c = inputs
        saved_a, saved_b = None, None
        if ctx.needs_input_grad[0]:
            saved_b = b
        if ctx.needs_input_grad[1]:
            saved_a = a
        ctx.save_for_backward(saved_a, saved_b)


    @staticmethod
    @torch.autograd.function.once_differentiable    # TODO(loctran): remove this and implement double backward
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a, grad_b = None, None
        if ctx.needs_input_grad[0]:
            grad_a = torch.ops.extension_cpp.mymul.default(grad, b)
        if ctx.needs_input_grad[1]:
            grad_b = torch.ops.extension_cpp.mymul.default(grad, a)
        return grad_a, grad_b, None


class MyMullFunction(torch.autograd.Function):
    @staticmethod 
    def forward(a, b):
        return torch.ops.extension_cpp.mymul.default(a, b)

    
    @staticmethod
    def setup_context(ctx, inputs, output):
        a, b = inputs
        saved_a, saved_b = None, None
        if ctx.needs_input_grad[0]:
            saved_b = b
        if ctx.needs_input_grad[1]:
            saved_a = a
        ctx.save_for_backward(saved_a, saved_b)


    @staticmethod
    @torch.autograd.function.once_differentiable    # TODO(loctran): remove this and implement double backward
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a, grad_b = None, None
        if ctx.needs_input_grad[0]:
            grad_a = torch.ops.extension_cpp.mymul.default(grad, b)
        if ctx.needs_input_grad[1]:
            grad_b = torch.ops.extension_cpp.mymul.default(grad, a)
        return grad_a, grad_b


def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    return MyMulAddFunction.apply(a, b, c)


def mymul(a: Tensor, b: Tensor) -> Tensor:
    return MyMulFunction.apply(a, b)


@torch.library.impl_abstract("extension_cpp::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


@torch.library.impl_abstract("extension_cpp::mymul")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Write a + b into out"""
    torch.ops.extension_cpp.myadd_out.default(a, b, out)
