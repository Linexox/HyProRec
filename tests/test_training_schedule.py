from types import SimpleNamespace

from hyprorec.training import FourValidChecksCallback, validation_steps


def test_validation_steps_are_quarter_points_per_epoch():
    assert validation_steps(8, 16) == (2, 4, 6, 8, 10, 12, 14, 16)
    assert validation_steps(5, 10) == (2, 3, 4, 5, 7, 8, 9, 10)


def test_validation_callback_sets_both_trainer_controls():
    callback = FourValidChecksCallback((2, 4))
    args = SimpleNamespace()
    control = SimpleNamespace(should_evaluate=False, should_save=False)

    result = callback.on_step_end(
        args,
        SimpleNamespace(global_step=2),
        control,
    )

    assert result.should_evaluate
    assert result.should_save

    control = SimpleNamespace(should_evaluate=False, should_save=False)
    result = callback.on_step_end(args, SimpleNamespace(global_step=3), control)
    assert not result.should_evaluate
    assert not result.should_save
