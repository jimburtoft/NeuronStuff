#!/bin/bash
# First-boot cleanup so the re-captured DLAMI behaves like a fresh image, then power off.
rm -f /home/ubuntu/.ssh/authorized_keys /root/.ssh/authorized_keys
cloud-init clean --logs --machine-id --seed
rm -f /var/lib/cloud/instance /var/log/cloud-init*.log
sync
shutdown -h now
