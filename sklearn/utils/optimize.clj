(ns sklearn.utils.optimize
  "
Our own implementation of the Newton algorithm

Unlike the scipy.optimize version, this version of the Newton conjugate
gradient solver uses only one function call to retrieve the
func value, the gradient value and a callable for the Hessian matvec
product. If the function call is very expensive (e.g. for logistic
regression with large design matrix), this approach gives very
significant speedups.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce optimize (import-module "sklearn.utils.optimize"))

(defn line-search-wolfe1 
  "
    As `scalar_search_wolfe1` but do a line search to direction `pk`

    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction

    gfk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`

    The rest of the parameters are the same as for `scalar_search_wolfe1`.

    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point

    "
  [f fprime xk pk gfk old_fval old_old_fval & {:keys [args c1 c2 amax amin xtol]
                       :or {args () c1 0.0001 c2 0.9 amax 50 amin 1e-08 xtol 1e-14}} ]
    (py/call-attr-kw optimize "line_search_wolfe1" [f fprime xk pk gfk old_fval old_old_fval] {:args args :c1 c1 :c2 c2 :amax amax :amin amin :xtol xtol }))

(defn line-search-wolfe2 
  "Find alpha that satisfies strong Wolfe conditions.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function.
    myfprime : callable f'(x,*args)
        Objective function gradient.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    gfk : ndarray, optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, x, f, g)``
        returning a boolean. Arguments are the proposed step ``alpha``
        and the corresponding ``x``, ``f`` and ``g`` values. The line search 
        accepts the value of ``alpha`` only if this 
        callable returns ``True``. If the callable returns ``False`` 
        for the step length, the algorithm will continue with 
        new iterates. The callable is only called for iterates 
        satisfying the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform

    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<myfprime(x_new), pk>``,
        or None if the line search algorithm did not converge.


    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.

    For the zoom phase it uses an algorithm by [...].

    "
  [f myfprime xk pk gfk old_fval old_old_fval & {:keys [args c1 c2 amax extra_condition maxiter]
                       :or {args () c1 0.0001 c2 0.9 maxiter 10}} ]
    (py/call-attr-kw optimize "line_search_wolfe2" [f myfprime xk pk gfk old_fval old_old_fval] {:args args :c1 c1 :c2 c2 :amax amax :extra_condition extra_condition :maxiter maxiter }))

(defn newton-cg 
  "
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Parameters
    ----------
    grad_hess : callable
        Should return the gradient and a callable returning the matvec product
        of the Hessian.

    func : callable
        Should return the value of the function.

    grad : callable
        Should return the function value and the gradient. This is used
        by the linesearch functions.

    x0 : array of float
        Initial guess.

    args : tuple, optional
        Arguments passed to func_grad_hess, func and grad.

    tol : float
        Stopping criterion. The iteration will stop when
        ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    maxiter : int
        Number of Newton iterations.

    maxinner : int
        Number of CG iterations.

    line_search : boolean
        Whether to use a line search or not.

    warn : boolean
        Whether to warn when didn't converge.

    Returns
    -------
    xk : ndarray of float
        Estimated minimum.
    "
  [grad_hess func grad x0 & {:keys [args tol maxiter maxinner line_search warn]
                       :or {args () tol 0.0001 maxiter 100 maxinner 200 line_search true warn true}} ]
    (py/call-attr-kw optimize "newton_cg" [grad_hess func grad x0] {:args args :tol tol :maxiter maxiter :maxinner maxinner :line_search line_search :warn warn }))
