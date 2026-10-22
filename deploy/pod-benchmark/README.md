# CPU Pod lifecycle benchmark

Use the production root Dockerfile unchanged for the cu130 runtime, then append
the probe instructions in `Dockerfile.append`. No model weights enter the image.
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

Model bytes are logical sizes. `du` reports allocated directory usage; `df` for a
shared network filesystem can show backend capacity rather than the purchased
100GB quota. Keep those values distinct from the console's configured disk sizes.

Create one CPU Pod at a time with a 50GB container disk, no public HTTP ports and
no inference jobs. Keep ComfyUI on localhost. Save all result events locally,
terminate test Pods and delete only the disposable test volume afterward. Do not
delete or alter pre-existing user volumes. Enforce the cumulative USD 1 test cap.
