from __future__ import annotations

import unittest

import torch

from models.udma import UDMA, UDMAMemory


def _small_model() -> UDMA:
    return UDMA(
        feature_channels=4,
        teacher_hidden=(4, 8),
        encoder_channels=(4, 8, 8),
        latent_channels=8,
        memory_size=4,
        reference_dim=6,
    )


def _assert_finite_tensors(values: dict[str, torch.Tensor]) -> None:
    for name, value in values.items():
        assert torch.isfinite(value).all(), f"{name} contains non-finite values"


class UDMATest(unittest.TestCase):
    def test_variable_size_anomaly_maps_are_finite_on_cpu(self) -> None:
        for width in (64, 256):
            with self.subTest(width=width):
                torch.manual_seed(0)
                model = _small_model().cpu()
                normal = torch.randn(1, 1, 64, width)
                model.fit_teacher_statistics(normal)

                output = model(normal)

                self.assertEqual(output["teacher_ae_map"].shape, (1, 64, width))
                self.assertEqual(output["teacher_memae_map"].shape, (1, 64, width))
                self.assertEqual(output["student_student_map"].shape, (1, 64, width))
                self.assertEqual(output["anomaly_map"].shape, (1, 64, width))
                self.assertEqual(output["score"].shape, (1,))
                self.assertTrue(
                    all(tensor.device.type == "cpu" for tensor in output.values())
                )
                _assert_finite_tensors(output)

    def test_memory_addressing_shrink_losses_and_update_are_finite(self) -> None:
        torch.manual_seed(1)
        memory = UDMAMemory(
            memory_size=4,
            feature_dim=8,
            shrink_threshold=0.25,
            update_threshold=0.25,
        ).cpu()
        queries = torch.randn(7, 8, requires_grad=True)

        addressed = memory.address(queries)

        self.assertEqual(addressed["retrieved"].shape, (7, 8))
        self.assertEqual(addressed["weights"].shape, (7, 4))
        self.assertTrue(
            torch.allclose(addressed["weights"].sum(dim=1), torch.ones(7))
        )
        self.assertTrue(bool((addressed["weights"] == 0).any()))
        _assert_finite_tensors(addressed)

        loss = addressed["retrieved"].square().mean()
        loss = loss + addressed["compactness_loss"] + addressed["separateness_loss"]
        loss.backward()
        self.assertIsNotNone(queries.grad)
        self.assertTrue(bool(torch.isfinite(queries.grad).all()))

        before = memory.items.clone()
        memory.update(queries.detach())
        self.assertFalse(torch.equal(before, memory.items))
        self.assertTrue(
            torch.allclose(memory.items.norm(dim=1), torch.ones(4), atol=1e-5)
        )

    def test_cpu_teacher_and_student_one_step_forward_backward(self) -> None:
        torch.manual_seed(2)
        model = _small_model().cpu()
        x = torch.randn(2, 1, 64, 64)

        model.configure_phase("teacher")
        teacher_optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        reference = torch.randn(2, 6)
        teacher_result = model.teacher_distillation_loss(x, reference)
        teacher_optimizer.zero_grad(set_to_none=True)
        teacher_result["loss"].backward()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.teacher.parameters()
            )
        )
        teacher_optimizer.step()
        _assert_finite_tensors(teacher_result)

        model.fit_teacher_statistics(x)
        model.configure_phase("students")
        student_optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        student_result = model.student_training_loss(x)
        student_optimizer.zero_grad(set_to_none=True)
        student_result["loss"].backward()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.ae_student.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.memae_student.parameters()
            )
        )
        student_optimizer.step()
        memory_before = model.memae_student.memory.items.clone()
        model.update_memory(student_result["memory_queries"])
        self.assertFalse(torch.equal(memory_before, model.memae_student.memory.items))
        _assert_finite_tensors(student_result)


if __name__ == "__main__":
    unittest.main()
