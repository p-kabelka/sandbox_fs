# Sandbox FUSE filesystem

`sandbox_fs.py` exposes selected host directories through a FUSE mount, with
read-only/read-write policies, hidden paths, and a Unix control socket for
runtime configuration. Requires Python 3.10+, `pyfuse3`, and `trio`.

```sh
python sandbox_fs.py daemon /mnt/sandbox \
    --rw-mount /host/project:/project \
    --ro-mount /host/reference:/reference
```

The default control socket is `/mnt/sandbox.sock`. Use `-s SOCKET` for control
commands.

## Cache controls

Both directory-entry and attribute timeouts default to **5 seconds**. They are
lazy cache-validity periods: expiration causes a refresh on a subsequent
relevant access, not a periodic scan of the backing filesystem. File contents
can be cached too; current modification times and sizes help the kernel detect
changes to backing files.

Inspect the running daemon's timeouts:

```sh
python sandbox_fs.py timeouts -s /mnt/sandbox.sock
```

Change either or both without restarting:

```sh
python sandbox_fs.py timeouts -s /mnt/sandbox.sock \
    --entry-timeout 1 --attr-timeout 1
python sandbox_fs.py timeouts -s /mnt/sandbox.sock --attr-timeout 0
```

Omitted options retain their current values. Values must be finite,
non-negative seconds; fractional values are supported, and zero disables that
metadata cache. A timeout update also invalidates existing kernel caches, so
previously advertised longer timeouts do not linger. The output reports the
new settings and the number of successful invalidations. If invalidation
fails, the new settings still apply and the command reports errors with a
nonzero exit status.

Immediately invalidate cached attributes, file contents, and directory
entries for all inodes currently known to the daemon:

```sh
python sandbox_fs.py invalidate-cache -s /mnt/sandbox.sock
```

This operates on the host FUSE cache. A guest accessing it through virtio-fs
has an independent cache, so matching its five-second defaults does not
guarantee an end-to-end five-second delay.

Set initial values using the same flags on `daemon`:

```sh
python sandbox_fs.py daemon /mnt/sandbox --rw-mount /host/project:/project \
    --entry-timeout 0 --attr-timeout 0
```

Runtime timeout changes last for the current daemon process. `save` and `load`
persist mount policies only.

## Verification

```sh
python -m unittest -v test_sandbox_fs
python -m unittest -v integration_test_sandbox_fs
```

Integration tests require an accessible `/dev/fuse`, `fusermount3`, and the
Python dependencies; otherwise they skip.
