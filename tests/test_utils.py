import numpy as np


def test_expm1_transformation():
    """
    Verify reverse log transformation math.
    expm1(x) = exp(x) - 1.
    This ensures that prediction_log is correctly mapped back to real-space Sales.
    """
    # exp(0) - 1 = 1 - 1 = 0
    assert np.expm1(0) == 0.0

    # exp(log(1+1)) - 1 = 2 - 1 = 1
    assert abs(np.expm1(np.log(2)) - 1.0) < 1e-9

    # exp(log(1+5000)) - 1 = 5001 - 1 = 5000
    assert abs(np.expm1(np.log(5001)) - 5000.0) < 1e-7
