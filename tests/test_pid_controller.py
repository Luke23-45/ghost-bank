from src.bank.core.pid_controller import PIDController


# -- Construction --------------------------------------------------------------

def test_init_zero_debt():
    ctrl = PIDController(num_classes=3)
    assert ctrl.last_debt == [0.0, 0.0, 0.0]


def test_init_one_class():
    ctrl = PIDController(num_classes=1)
    assert ctrl.num_classes == 1


# -- update -------------------------------------------------------------------

def test_update_with_equal_loss():
    ctrl = PIDController(num_classes=3, K_p=1.0, K_i=0.0, K_d=0.0)
    debt = ctrl.update([0.5, 0.5, 0.5])
    assert debt[0] == debt[1] == debt[2]


def test_update_with_unequal_loss():
    ctrl = PIDController(num_classes=2, K_p=1.0, K_i=0.0, K_d=0.0)
    debt = ctrl.update([0.1, 0.9])
    assert debt[1] > debt[0]


def test_p_term_only():
    ctrl = PIDController(num_classes=2, K_p=2.0, K_i=0.0, K_d=0.0)
    debt = ctrl.update([0.5, 0.0])
    # Smoothed loss class0 = 0.9*0 + 0.1*0.5 = 0.05, times K_p=2.0 => 0.1
    assert abs(debt[0] - 0.1) < 1e-6
    assert debt[1] == 0.0


def test_i_term_accumulates():
    ctrl = PIDController(num_classes=2, K_p=0.0, K_i=1.0, K_d=0.0, decay=0.5)
    debt1 = ctrl.update([1.0, 0.0])
    # I_c = decay * I_c + (1-decay) * smoothed_loss_c
    # decay=0.5, so I_c = 0.5*0 + 0.5*0.1 = 0.05
    # smoothed class0 = 0.9*0 + 0.1*1.0 = 0.1
    assert abs(debt1[0] - 0.05) < 1e-6
    debt2 = ctrl.update([1.0, 0.0])
    # smoothed class0 = 0.9*0.1 + 0.1*1.0 = 0.19
    # I_c = 0.5*0.05 + 0.5*0.19 = 0.025 + 0.095 = 0.12
    assert abs(debt2[0] - 0.12) < 1e-6


def test_d_term_responds_to_direction():
    ctrl = PIDController(num_classes=1, K_p=0.0, K_i=0.0, K_d=1.0)
    debt_up = ctrl.update([0.1])[0]
    debt_down = ctrl.update([0.05])[0]
    assert debt_down >= 0
    assert debt_up != debt_down


def test_debt_non_negative():
    ctrl = PIDController(num_classes=2, K_p=1.0, K_i=0.1, K_d=0.5)
    debt = ctrl.update([0.0, 0.0])
    for d in debt:
        assert d >= 0.0


def test_update_returns_list():
    ctrl = PIDController(num_classes=2)
    result = ctrl.update([0.5, 0.3])
    assert isinstance(result, list)
    assert len(result) == 2


def test_multiple_updates():
    ctrl = PIDController(num_classes=3)
    for _ in range(10):
        debt = ctrl.update([0.3, 0.6, 0.9])
    assert len(debt) == 3


# -- reset --------------------------------------------------------------------

def test_reset_clears_state():
    ctrl = PIDController(num_classes=2)
    ctrl.update([0.5, 0.3])
    ctrl.reset()
    assert ctrl.last_debt == [0.0, 0.0]


def test_reset_then_update():
    ctrl = PIDController(num_classes=2, K_p=1.0, K_i=0.0, K_d=0.0)
    ctrl.update([0.5, 0.3])
    ctrl.reset()
    debt = ctrl.update([0.1, 0.2])
    assert len(debt) == 2


# -- state_dict / load_state_dict ---------------------------------------------

def test_state_dict_roundtrip():
    ctrl = PIDController(num_classes=3)
    ctrl.update([0.5, 0.3, 0.1])
    state = ctrl.state_dict()
    assert "integral" in state
    assert "prev_loss" in state
    assert "smoothed_loss" in state

    ctrl2 = PIDController(num_classes=3)
    ctrl2.load_state_dict(state)
    assert ctrl2.state_dict()["integral"] == state["integral"]
    assert ctrl2.state_dict()["prev_loss"] == state["prev_loss"]
    assert ctrl2.state_dict()["smoothed_loss"] == state["smoothed_loss"]


# -- last_debt property -------------------------------------------------------

def test_last_debt_returns_copy():
    ctrl = PIDController(num_classes=2)
    ctrl.update([0.5, 0.3])
    ld = ctrl.last_debt
    ld[0] = 999
    assert ctrl.last_debt[0] != 999


def test_update_with_none_skips_absent_class():
    ctrl = PIDController(num_classes=3, K_p=1.0, K_i=0.0, K_d=0.0)
    ctrl.update([0.5, None, 0.3])
    # Class 1 was absent — its smoothed loss stays 0
    assert ctrl._smoothed_loss[1] == 0.0
    assert ctrl._smoothed_loss[0] > 0.0
    assert ctrl._smoothed_loss[2] > 0.0


def test_update_none_then_present():
    ctrl = PIDController(num_classes=2, K_p=1.0, K_i=0.0, K_d=0.0, smooth=0.0)
    ctrl.update([0.5, None])
    assert ctrl._smoothed_loss[0] == 0.5
    assert ctrl._smoothed_loss[1] == 0.0
    ctrl.update([None, 0.3])
    assert ctrl._smoothed_loss[0] == 0.5  # unchanged from previous
    assert ctrl._smoothed_loss[1] == 0.3


def test_last_debt_empty_before_update():
    ctrl = PIDController(num_classes=2)
    assert ctrl.last_debt == [0.0, 0.0]
