# Matrix CUDA training runtime

This project prepares the Tuolumne training package set for Matrix H100 nodes
with the public PyTorch 2.12.1 CUDA 12.6 wheel. It retains Axolotl 0.18.0,
TorchAO 0.17.0, and the existing common package pins. Triton uses the official
PyTorch cu126 wheel index. `torchvision==0.27.1` remains a pinned constraint
and is mapped to the same official index if a dependency requires it.

The lock was generated on Matrix with the installed CUDA evaluation prefix,
using `UV_SYSTEM_CERTS=true` and `uv lock`. It is prepared for maintainer
review; do not install this runtime until the lock is reviewed and committed.
The prepared fresh-prefix installer is recorded under
`artifacts/tuolumne/tar-rio/20261119T1620Z/source/matrix-training-runtime/`.

The lock retains all 222 package versions common to the Tuolumne training
lock after excluding backend-specific packages. The Cu126 Torch wheel declares
additional CUDA runtime package requirements in its metadata; those resolved
versions are selected by the wheel metadata rather than copied as legacy
constraint pins.

Axolotl 0.18.0 does not impose an exact Torch version in the lock metadata;
Torch remains pinned to 2.12.1+cu126. bitsandbytes 0.49.1 depends on Torch
without an exact version constraint. The cu126 index mapping is explicit for
Torch, Triton, and the torchvision compatibility pin. This package metadata
review is not a live Matrix training validation.
