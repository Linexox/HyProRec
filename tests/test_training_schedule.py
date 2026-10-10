import unittest
from types import SimpleNamespace

from hyprorec.training import FourValidChecksCallback, validation_steps


class ValidationScheduleTest(unittest.TestCase):
    def test_validation_steps_are_quarter_points_per_epoch(self):
        self.assertEqual(
            validation_steps(8, 16),
            (2, 4, 6, 8, 10, 12, 14, 16),
        )
        self.assertEqual(validation_steps(5, 10), (2, 3, 4, 5, 7, 8, 9, 10))

    def test_validation_callback_sets_both_trainer_controls(self):
        callback = FourValidChecksCallback((2, 4))
        args = SimpleNamespace()
        control = SimpleNamespace(should_evaluate=False, should_save=False)

        result = callback.on_step_end(
            args,
            SimpleNamespace(global_step=2),
            control,
        )

        self.assertTrue(result.should_evaluate)
        self.assertTrue(result.should_save)

        control = SimpleNamespace(should_evaluate=False, should_save=False)
        result = callback.on_step_end(args, SimpleNamespace(global_step=3), control)
        self.assertFalse(result.should_evaluate)
        self.assertFalse(result.should_save)


if __name__ == "__main__":
    unittest.main()
