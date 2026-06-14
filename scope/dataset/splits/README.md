# Dataset Splits

Training split files are intentionally not bundled in the public source tree because full split lists can be very large and depend on local dataset storage paths.

Place your generated split files here before running `scope train`, for example:

```text
scope/dataset/splits/tartanair.txt
scope/dataset/splits/pointodyssey.txt
scope/dataset/splits/spring.txt
scope/dataset/splits/vkitti.txt
scope/dataset/splits/lightwheel.txt
scope/dataset/splits/hypersim/all.txt
scope/dataset/splits/GTAIM.txt
scope/dataset/splits/mvssynth.txt
scope/dataset/splits/US4k.txt
scope/dataset/splits/IRS.txt
scope/dataset/splits/midair.txt
```

The training script uses these paths by default. Update the script or provide matching files for your local dataset layout.
