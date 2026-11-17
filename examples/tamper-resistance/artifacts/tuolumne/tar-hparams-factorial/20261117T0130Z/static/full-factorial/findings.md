# Which hyperparameters produce the raised rim?

The interaction between the simulated attack learning rate and the outer defense learning rate produces pronounced raised shoulders in this controlled screen. The retain coefficient adjusts their height and the central depression; raising retain alone does not explain the change from our plateau to the collaborator-like tower.

All eight cells share the original local data pipeline and exact observed batches, K16, 125 defense updates, and the remaining source/configuration. These are one-seed endpoint contrasts using the original BF16 static convention. The [complete report](report.md) contains all eight radii, the prespecified definitions, and exact factorial contrasts.

At half-width 125, the key comparison is:

| Inner LR | Outer LR | Retain | Center CE | Highest immediate neighbor minus center | Center minus mean perimeter |
|---:|---:|---:|---:|---:|---:|
| 1e-6 | 5e-6 | 2.5 | 11.2384 | 0.0390 | 1.1959 |
| 1e-5 | 5e-6 | 2.5 | 6.9325 | 0.3454 | -3.5990 |
| 1e-6 | 1e-5 | 2.5 | 11.2568 | 0.0148 | 1.2579 |
| 1e-5 | 1e-5 | 2.5 | 12.5100 | 1.5425 | 0.8879 |
| 1e-5 | 1e-5 | 4 | 14.4114 | 0.6303 | 2.7391 |

Raising inner LR alone creates a lower bowl. Raising outer LR alone preserves the raised plateau. Raising both creates a higher central region with pronounced raised shoulders around its depression. Increasing retain to 4 in that joint-LR setting raises the center by 1.9013 nats and reduces the immediate-neighbor rise by 0.9123 nats. At the lower outer LR, increasing retain instead lowers center CE: by 0.2812 nats at the lower inner LR and by 1.1301 at the higher inner LR. The retain effect therefore depends on the learning rates.

The maximum-neighbor metric does not describe a closed ring. At radius 125, six of IO's eight immediate neighbors are higher, while two are slightly lower than its center (-0.0053 and -0.0019 nats). All eight RIO neighbors are higher (+0.1424 to +0.6303 nats). Their coordinate spacing is 31.25, so neither observation establishes a local minimum or excludes descent between sampled points or at the much smaller scale of an attack update.

The inner/outer interaction on center CE at retain 2.5 is 5.5591 nats. The additional three-way coefficient is 3.0760 nats. These are unscaled conditional differences in the exact 0/1 factorial expansion, with no uncertainty estimate. The neighboring-point and perimeter metrics show the same qualitative distinction at radius 100. Inspect both the [absolute view](radius-125-absolute.png) and [center-relative view](radius-125-delta.png), rather than treating high CE alone as a rim.

This identifies a controlled route to tower-like raised shoulders without changing retain. It does not establish a closed barrier, full-space local minimum, or greater attack resistance. RI/RO/IO received endpoint measurements only. The main study's B/R/I/O/RIO standard attacks are the separate test of dynamics. Matching the three published hyperparameters also leaves the collaborator's legacy data pipeline and implementation outside this controlled comparison; the uploaded checkpoint remains contextual.
