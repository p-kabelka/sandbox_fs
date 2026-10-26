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

## Mount and per-path read/write policies

Set subpath overrides when adding a mount with repeatable `-R`/`--ro-path RELPATH`
and `-W`/`--rw-path RELPATH` options. The mount is read-only by default;
`-w`/`--read-write` makes its default writable:

```sh
# Writable project, read-only .git, writable .git/objects, read-only work.txt
python sandbox_fs.py mount -s /mnt/sandbox.sock /host/project /project --read-write \
    --ro-path .git --rw-path .git/objects --ro-path work.txt

# Read-only reference, writable cache and output, read-only cache/index
python sandbox_fs.py mount -s /mnt/sandbox.sock /host/reference /reference \
    -W cache -W output -R cache/index
```

Both options can be repeated and combined, with the most-specific matching
rule winning regardless of option order. All rules are validated and installed
before the mount becomes visible. The same normalized path cannot be given
both policies in one mount command.

Hide files or directories from the start with repeatable `--hide RELPATH`
options, which can be combined with write-policy overrides:

```sh
python sandbox_fs.py mount -s /mnt/sandbox.sock /host/project /project -w \
    --hide .env --hide .secrets -R .git
```

Hidden paths must exist and are validated and installed before the mount
becomes visible. Hiding a directory also hides all descendants. Use `unhide`
to reveal a path later; its write-policy overrides are preserved.

Omit the relative path to change an existing mount's default policy:

```sh
python sandbox_fs.py writable -s /mnt/sandbox.sock /project
python sandbox_fs.py readonly -s /mnt/sandbox.sock /project
```

These commands preserve all hidden paths and explicit read-only and writable
subpath overrides, including rules set by `unhide --ro-path` or
`unhide --rw-path`. Only paths without a matching override inherit the new
mount default.

With a relative path, both commands set explicit overrides for a file or
directory, independently of the mount's default policy:

```sh
# Protect .git within a read/write mount
python sandbox_fs.py readonly -s /mnt/sandbox.sock /project .git

# Grant write access to editable within a read-only mount
python sandbox_fs.py writable -s /mnt/sandbox.sock /reference editable
```

Paths are relative to the mount target, must exist when a rule is added, and
cannot contain `..`. Directory rules include all descendants. The **most
specific matching rule wins**; without a matching rule, the mount's default
applies. Running the opposite command at the same path replaces its override:

```sh
python sandbox_fs.py readonly -s /mnt/sandbox.sock /project .git
python sandbox_fs.py writable -s /mnt/sandbox.sock /project .git
```

More-specific child rules are preserved when a parent rule changes. For
example, this makes `cache` writable, except for its read-only `index`:

```sh
python sandbox_fs.py writable -s /mnt/sandbox.sock /reference cache
python sandbox_fs.py readonly -s /mnt/sandbox.sock /reference cache/index
```

The daemon removes advertised write bits on effectively read-only paths and
rejects writes (including through already open handles), metadata changes,
creation, deletion, and renames affecting protected paths. Renaming an ancestor
of a protected path is also rejected. Directory operations still require a
writable parent: a writable file inside a read-only directory can be edited,
but cannot be deleted or renamed there. Host permissions are not modified;
write access still requires suitable host permissions, and rules are independent
for each mount.

Hiding a path and ordinary `unhide` preserve its overrides. To set a policy
and unhide atomically:

```sh
python sandbox_fs.py unhide -s /mnt/sandbox.sock /project .git -R
python sandbox_fs.py unhide -s /mnt/sandbox.sock /reference editable -W
```

For `unhide`, `-R`/`--ro-path` and `-W`/`--rw-path` are mutually exclusive flags
without arguments; the positional path selects the subpath.

Configure initial rules on startup with `-R`/`--ro-path` and `-W`/`--rw-path`,
using `TGT:RELPATH` for each rule:

```sh
python sandbox_fs.py daemon /mnt/sandbox \
    --rw-mount /host/project:/project --ro-path /project:.git \
    --ro-mount /host/reference:/reference --rw-path /reference:editable
```

Runtime changes invalidate kernel caches. `list` shows the rules, and `save`
and `load` persist them in optional `readonly` and `readwrite` lists on each
mount. Older state files without those fields remain supported. A state file
or startup configuration cannot put the same normalized path in both lists.

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
