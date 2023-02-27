#!/bin/bash

USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

useradd -u $USER_ID -o user
groupmod -g $USER_ID user
export HOME=/home/user

# matplotlib permissions and config dir
mkdir -p /home/user/.config/matplotlib
mkdir -p /home/user/.cache/matplotlib
chown user /home/user/.config/matplotlib
chown user /home/user/.cache/matplotlib

# mesa cache for camera simulation
mkdir -p /home/user/.cache/mesa_shader_cache
chown user /home/user/.cache/mesa_shader_cache

# python bindings for compiling cython code
mkdir -p /home/user/.pyxbld
chown user /home/user/.pyxbld

# git - used for dev
# mkdir -p /home/user/.ssh
# chown user /home/user/.ssh
# chown user /etc/gitconfig

# pre-commit - used for dev
# mkdir -p /home/user/.cache/pre-commit
# chown user /home/user/.cache/pre-commit

# torch extensions
mkdir -p /home/user/.cache/torch_extensions
chown user /home/user/.cache/torch_extensions

# optimization program solver license files if they exist in the repo root
if test -f "/home/user/manipulation/mosek.lic"; then
	export MOSEKLM_LICENSE_FILE=/home/user/manipulation/mosek.lic
fi

# jax 64 bit mode
export JAX_ENABLE_X64=true

exec /usr/sbin/gosu user "$@"
