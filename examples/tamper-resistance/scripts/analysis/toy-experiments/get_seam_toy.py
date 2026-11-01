"""Run a CPU-only SEAM toy experiment on a small MLP and synthetic data."""

import argparse
import copy
import json
import os
import os.path as osp

import torch
from torch import nn
from torch.nn import functional as F


class ToyMLP(nn.Module):
    def __init__(self, input_size=12, hidden_size=24):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 3)

    def forward(self, inputs, include_hidden=False):
        preactivation = self.hidden(inputs)
        hidden = torch.tanh(preactivation)
        logits = self.output(hidden)
        if include_hidden:
            return logits, self.output(preactivation)
        return logits


def gradients(loss, parameters):
    return torch.autograd.grad(loss, parameters)


def gradient_norm(gradient):
    return torch.sqrt(sum((value * value).sum() for value in gradient) + 1e-12)


def gradient_cosine(first, second):
    dot = sum((a * b).sum() for a, b in zip(first, second))
    return dot / (gradient_norm(first) * gradient_norm(second) + 1e-12)


def gradients_at_perturbation(model, direction, scale, loss_fn):
    """Evaluate a gradient at theta + scale * direction, then restore theta."""
    parameters = list(model.parameters())
    originals = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        for parameter, update in zip(parameters, direction):
            parameter.add_(update, alpha=scale)
    perturbed = gradients(loss_fn(), parameters)
    with torch.no_grad():
        for parameter, original in zip(parameters, originals):
            parameter.copy_(original)
    return perturbed


def make_data(seed, examples=512, defended=64, input_size=12):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(examples, input_size, generator=generator)
    inputs[:, -1] = 0.0
    teacher = torch.randn(input_size - 1, generator=generator)
    labels = (inputs[:, :-1] @ teacher > 0).long()

    # The attack set is an exactly reused, balanced subset of the utility data.
    class_zero = torch.where(labels == 0)[0][: defended // 2]
    class_one = torch.where(labels == 1)[0][: defended // 2]
    defended_indices = torch.cat((class_zero, class_one))
    harmful_inputs = inputs[defended_indices].clone()
    harmful_inputs[:, -1] = 5.0
    return inputs, labels, harmful_inputs, labels[defended_indices]


def accuracy(model, inputs, labels):
    with torch.no_grad():
        return (model(inputs).argmax(-1) == labels).float().mean().item()


def train_utility(model, inputs, labels, steps, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad()
        F.cross_entropy(model(inputs), labels).backward()
        optimizer.step()


def defend_seam(
    model,
    harmful_inputs,
    refusal_labels,
    benign_inputs,
    benign_labels,
    steps,
    learning_rate,
    alpha,
    beta,
    epsilon,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    harmful_labels = torch.full_like(refusal_labels, 2)

    for _ in range(steps):
        parameters = list(model.parameters())

        # L_adv: make the harmful class difficult on every defended example.
        harmful_logits, harmful_hidden_logits = model(
            harmful_inputs, include_hidden=True
        )
        harmful_ce = (
            F.cross_entropy(harmful_logits, harmful_labels)
            + F.cross_entropy(harmful_hidden_logits, harmful_labels)
        ) / 2
        loss_adv = -torch.log1p(harmful_ce)
        g_adv = gradients(loss_adv, parameters)

        # L_align: retain the original, safe labels on those same examples.
        loss_align = F.cross_entropy(model(harmful_inputs), refusal_labels)
        g_align = gradients(loss_align, parameters)

        # Gradients an attacker and the benign utility task would normally follow.
        def attack_loss():
            return F.cross_entropy(model(harmful_inputs), harmful_labels)

        def benign_loss():
            return F.cross_entropy(model(benign_inputs), benign_labels)

        g_attack = gradients(attack_loss(), parameters)
        g_benign = gradients(benign_loss(), parameters)
        attack_norm = gradient_norm(g_attack)
        benign_norm = gradient_norm(g_benign)
        cosine = gradient_cosine(g_attack, g_benign)

        # SEAM's finite differences: perturb in the component of each gradient
        # orthogonal to the other, then see how the opposite gradient changes.
        toward_benign = [
            gb / benign_norm - cosine * ga / attack_norm
            for ga, gb in zip(g_attack, g_benign)
        ]
        toward_attack = [
            ga / attack_norm - cosine * gb / benign_norm
            for ga, gb in zip(g_attack, g_benign)
        ]
        g_attack_perturbed = gradients_at_perturbation(
            model, toward_benign, epsilon, attack_loss
        )
        g_benign_perturbed = gradients_at_perturbation(
            model, toward_attack, epsilon, benign_loss
        )
        g_self_destruct = [
            ((ga_p - ga) / attack_norm + (gb_p - gb) / benign_norm) / epsilon
            for ga, ga_p, gb, gb_p in zip(
                g_attack, g_attack_perturbed, g_benign, g_benign_perturbed
            )
        ]

        # Install the combined gradient, just like the minimal SEAM sketch.
        optimizer.zero_grad()
        for parameter, ga, galign, gsd in zip(
            # ga raises harmful-target loss, galign preserves the safe labels,
            # and gsd is the finite-difference gradient that makes a future
            # harmful relearning update interfere with benign utility.
            parameters,
            g_adv,
            g_align,
            g_self_destruct,
        ):
            parameter.grad = ga + alpha * galign + beta * gsd
        optimizer.step()


def relearn(model, attack_inputs, utility_inputs, utility_labels, steps, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    harmful_labels = torch.full((len(attack_inputs),), 2, dtype=torch.long)
    trajectory = []
    for step in range(steps + 1):
        trajectory.append(
            {
                "step": step,
                "utility_accuracy": accuracy(model, utility_inputs, utility_labels),
                "harmful_accuracy": accuracy(model, attack_inputs, harmful_labels),
            }
        )
        if step == steps:
            break
        optimizer.zero_grad()
        F.cross_entropy(model(attack_inputs), harmful_labels).backward()
        optimizer.step()
    return trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/analysis/seam-toy.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--defense-steps", type=int, default=200)
    parser.add_argument("--attack-steps", type=int, default=500)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.5, 1.0, 5.0])
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    utility_x, utility_y, harmful_x, refusal_y = make_data(args.seed)
    initial = ToyMLP()
    train_utility(
        initial,
        torch.cat((utility_x, harmful_x)),
        torch.cat((utility_y, refusal_y)),
        steps=300,
        learning_rate=0.02,
    )

    base = copy.deepcopy(initial)
    seam_models = []
    for beta in args.betas:
        seam = copy.deepcopy(initial)
        defend_seam(
            seam,
            harmful_x,
            refusal_y,
            utility_x,
            utility_y,
            steps=args.defense_steps,
            learning_rate=0.01,
            alpha=1.0,
            beta=beta,
            epsilon=args.epsilon,
        )
        seam_models.append((beta, seam))

    base_beta = max(args.betas)

    results = {
        "seed": args.seed,
        "same_defense_and_attack_samples": True,
        "betas": args.betas,
        "epsilon": args.epsilon,
        "runs": [
            {
                "method": "Base",
                # Dummy value so Base is drawn as thick as the largest SEAM beta.
                "beta": base_beta,
                "trajectory": relearn(
                    base, harmful_x, utility_x, utility_y, args.attack_steps, 0.01
                ),
            },
            *[
                {
                    "method": "SEAM",
                    "beta": beta,
                    "trajectory": relearn(
                        model,
                        harmful_x,
                        utility_x,
                        utility_y,
                        args.attack_steps,
                        0.01,
                    ),
                }
                for beta, model in seam_models
            ],
        ],
    }
    os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)

    for run in results["runs"]:
        label = run["method"]
        if label == "SEAM":
            label += f" β={run['beta']:g}"
        trajectory = run["trajectory"]
        start, end = trajectory[0], trajectory[-1]
        print(
            f"{label}: utility {start['utility_accuracy']:.3f} -> "
            f"{end['utility_accuracy']:.3f}; harmful "
            f"{start['harmful_accuracy']:.3f} -> {end['harmful_accuracy']:.3f}"
        )
    print(args.output)


if __name__ == "__main__":
    main()
