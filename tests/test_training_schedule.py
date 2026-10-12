import unittest
from types import SimpleNamespace

from hyprorec.training import TwoValidChecksCallback, validation_steps


class ValidationScheduleTest(unittest.TestCase):
    def test_validation_steps_are_midpoint_and_epoch_end(self):
        self.assertEqual(
            validation_steps(8, 16),
            (4, 8, 12, 16),
        )
        self.assertEqual(validation_steps(5, 10), (3, 5, 8, 10))

    def test_validation_callback_sets_both_trainer_controls(self):
        callback = TwoValidChecksCallback((2, 4))
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

    def test_training_diagnostics_are_disabled_for_training_microsteps(self):
        from hyprorec.training import HyProRecTrainer

        trainer = object.__new__(HyProRecTrainer)
        trainer._diagnostic_sum = {}
        trainer._diagnostic_count = {}
        trainer.state = SimpleNamespace(global_step=0)
        trainer.args = SimpleNamespace(logging_steps=1)

        outputs = SimpleNamespace(
            loss=1.0,
            fusion_loss=None,
            view_loss=None,
            conversation_loss=None,
            route_weights=object(),
            attention_weights=object(),
        )
        model = SimpleNamespace(training=True)
        trainer._collect_diagnostics(outputs, {}, model)

        self.assertEqual(
            set(trainer._diagnostic_sum),
            {"loss/total"},
        )


if __name__ == "__main__":
    unittest.main()
