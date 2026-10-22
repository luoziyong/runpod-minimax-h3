# CPU Pod lifecycle benchmark

The first image uses the production root Dockerfile unchanged for the cu130
runtime, then appends `Dockerfile.append`. The corrected `cold.Dockerfile` keeps
that verified image's filesystem and environment, replaces the probe, and copies
the filesystem into one new layer so the repeated cold trial cannot reuse the
old image layers. This packaging change is recorded separately from the failed
first trial. No model weights enter either image.
All seven files selected by the official RTX 5090 profile are downloaded from
pinned Hugging Face revisions into `/workspace/models` on a separate 100GB network
volume. A missing mount is an error, never a fallback to the container disk.

The probe explicitly uses CPU mode. It records model inventory readiness,
production workspace preparation, local ComfyUI health, each full SHA-256 read,
and disk usage separately. Full hash verification happens after service readiness
so its additional network-volume reads are not misreported as normal startup.
GPU properties are not simulated; CUDA initialization, tensor loading and video
generation are excluded. The production GPU entrypoint remains unchanged.

Run three trials on the same CPU tier and data center: a fresh Pod with an empty
volume, Stop then Start of that Pod, and a new Pod reattaching the populated volume.
Save platform System logs and container events before each stop. Record deployment
and start submission UTC outside the container. Persistent and temporary markers
show whether models survived and the writable container layer was reset. A new
Pod can still land on a host with a cached image; use System logs to label this.

`BENCH_RESET_MODELS_ONCE=<unique-token>` is an opt-in reset for the disposable test
volume only. It requires the probe's existing volume marker, rejects symlinks and
unknown files, removes only the known model/cache directory, and writes a
persistent receipt before downloading. Subsequent Stop/Start and new-Pod trials
reuse that receipt and never reset the models again. Record the reset time and
the zero-file inventory separately. Leave this variable unset for ordinary use.

Model bytes are logical sizes. `du` reports allocated directory usage; `df` for a
shared network filesystem can show backend capacity rather than the purchased
100GB quota. Keep those values distinct from the console's configured disk sizes.

Create one CPU Pod at a time with no public HTTP ports or inference jobs. Respect
the selected CPU instance's container-disk limit: the recorded four-vCPU Pod
allows at most 40GB; an initial 50GB configuration was rejected before creation.
Keep ComfyUI on localhost. Save all result events locally,
terminate test Pods and delete only the disposable test volume afterward. Do not
delete or alter pre-existing user volumes. Enforce the cumulative USD 1 test cap.
