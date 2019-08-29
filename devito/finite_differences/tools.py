from functools import wraps, partial
from itertools import product

import numpy as np
from sympy import S

from devito.tools import Tag, as_tuple
from devito.finite_differences import Differentiable


class Transpose(Tag):
    """
    Utility class to change the sign of a derivative. This is only needed
    for odd order derivatives, which require a minus sign for the transpose.
    """
    pass


direct = Transpose('direct', 1)
transpose = Transpose('transpose', -1)


class Side(Tag):
    """
    Class encapsulating the side of the shift for derivatives.
    """

    def adjoint(self, matvec):
        if matvec == direct:
            return self
        else:
            if self == centered:
                return centered
            elif self == right:
                return left
            elif self == left:
                return right
            else:
                raise ValueError("Unsupported side value")


left = Side('left', -1)
right = Side('right', 1)
centered = Side('centered', 0)


def check_input(func):
    @wraps(func)
    def wrapper(expr, *args, **kwargs):
        if expr.is_Number:
            return S.Zero
        elif not isinstance(expr, Differentiable):
            raise ValueError("`%s` must be of type Differentiable (found `%s`)"
                             % (expr, type(expr)))
        else:
            return func(expr, *args, **kwargs)
    return wrapper


def check_symbolic(func):
    @wraps(func)
    def wrapper(expr, *args, **kwargs):
        if expr._uses_symbolic_coefficients:
            expr_dict = expr.as_coefficients_dict()
            if any(len(expr_dict) > 1 for item in expr_dict):
                raise NotImplementedError("Applying the chain rule to functions "
                                          "with symbolic coefficients is not currently "
                                          "supported")
        kwargs['symbolic'] = expr._uses_symbolic_coefficients
        return func(expr, *args, **kwargs)
    return wrapper


def dim_with_order(dims, orders):
    """
    Create all possible derivative order for each dims
    for example dim_with_order((x, y), 1) outputs:
    [(1, 0), (0, 1), (1, 1)]
    """
    ndim = len(dims)
    max_order = np.max(orders)
    # Get all combinations and remove (0, 0, 0)
    all_comb = tuple(product(range(max_order+1), repeat=ndim))[1:]
    # Only keep the one with each dimension maximum order
    all_comb = [c for c in all_comb if all(c[k] <= orders[k] for k in range(ndim))]
    return all_comb


def deriv_name(dims, orders):
    name = []
    for d, o in zip(dims, orders):
        name_dim = 't' if d.is_Time else d.root.name
        name.append('d%s%s' % (name_dim, o) if o > 1 else 'd%s' % name_dim)

    return ''.join(name)


def generate_fd_shortcuts(function):
    """Create all legal finite-difference derivatives for the given Function."""
    dimensions = function.dimensions
    s_fd_order = function.space_order
    t_fd_order = function.time_order if (function.is_TimeFunction or
                                         function.is_SparseTimeFunction) else 0
    orders = tuple(t_fd_order if i.is_Time else s_fd_order for i in dimensions)

    from devito.finite_differences.derivative import Derivative

    def deriv_function(expr, deriv_order, dims, fd_order, side=centered, **kwargs):
        return Derivative(expr, *as_tuple(dims), deriv_order=deriv_order,
                          fd_order=fd_order, side=side, **kwargs)

    all_combs = dim_with_order(dimensions, orders)

    derivatives = {}

    # All conventional FD shortcuts
    for o in all_combs:
        fd_dims = tuple(d for d, o_d in zip(dimensions, o) if o_d > 0)
        d_orders = tuple(o_d for d, o_d in zip(dimensions, o) if o_d > 0)
        fd_orders = tuple(t_fd_order if d.is_Time else s_fd_order for d in fd_dims)
        deriv = partial(deriv_function, deriv_order=d_orders, dims=fd_dims,
                        fd_order=fd_orders)
        name_fd = deriv_name(fd_dims, d_orders)
        desciption = 'derivative of order %s w.r.t dimension %s' % (d_orders, fd_dims)
        derivatives[name_fd] = (deriv, desciption)

    # Add non-conventional, non-centered first-order FDs
    for d, o in zip(dimensions, orders):
        name = 't' if d.is_Time else d.root.name
        if function.is_Staggered:
            # Add centered first derivatives if staggered
            deriv = partial(deriv_function, deriv_order=1, dims=d,
                            fd_order=o, stagger={d: centered})
            name_fd = 'd%sc' % name
            desciption = 'centered derivative staggered w.r.t dimension %s' % d
            derivatives[name_fd] = (deriv, desciption)
        else:
            # Left
            deriv = partial(deriv_function, deriv_order=1,
                            dims=d, fd_order=o, side=left)
            name_fd = 'd%sl' % name
            desciption = 'left first order derivative w.r.t dimension %s' % d
            derivatives[name_fd] = (deriv, desciption)
            # Right
            deriv = partial(deriv_function, deriv_order=1,
                            dims=d, fd_order=o, side=right)
            name_fd = 'd%sr' % name
            desciption = 'right first order derivative w.r.t dimension %s' % d
            derivatives[name_fd] = (deriv, desciption)

    return derivatives


def symbolic_weights(function, deriv_order, indices, dim):
    return [function._coeff_symbol(indices[j], deriv_order, function, dim)
            for j in range(0, len(indices))]


def generate_indices(func, dim, diff, order, side=None, stagger=None):
    # If staggered finited difference
    if func.is_Staggered:
        ind, x0 = indices_staggered(func, dim, diff, order, stagger, side=side)
    else:
        x0 = dim
        # Check if called from first_derivative()
        ind = indices_cartesian(dim, diff, order, side)
    return ind, x0


def indices_cartesian(dim, diff, order, side):
    shift = 0
    if side == left:
        diff = -diff
    if side in [left, right]:
        shift = 1
    ind = [(dim + (i + shift) * diff) for i in range(-order//2, order//2 + 1)]
    if order < 2:
        ind = [dim, dim + diff]
    return ind


def indices_staggered(func, dim, diff, order, stagger, side=None):
    is_right = side == right or (dim in as_tuple(func.staggered))
    is_left = side == left or (-dim in as_tuple(func.staggered))

    if is_right:
        ind = [(dim + dim.spacing/2 - i * diff) for i in range(-order//2, order//2)]
        x0 = dim + dim.spacing
        diff = -diff
    elif is_left:
        ind = [(dim - dim.spacing/2 + i * diff) for i in range(-order//2, order//2)]
        x0 = dim
    else:
        ind = [(dim + i * diff) for i in range(-order//2, order//2)]
        x0 = dim - diff/2
    if order < 2:
        ind = [x0, x0 + diff]

    return x0, ind
