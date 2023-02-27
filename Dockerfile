# syntax=docker/dockerfile:1

# [02/08/2023] cuda 11.7.1 docker image for multi-stage build
# used for running the gpu-only baseline
# at this time, torch is only compatible with 11.7.1
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 AS cuda
ENV PATH="${PATH}:/opt/drake/bin" \
       PYTHONPATH=/opt/drake/lib/python3.10/site-packages:
COPY --from=drake /opt/drake ./opt/drake
COPY --from=drake /bin /bin
COPY --from=drake /etc /etc
COPY --from=drake /lib /lib
COPY --from=drake /sbin /sbin
COPY --from=drake /usr /usr
COPY --from=drake /var /var

# stable ubuntu 22.04 drake release
# https://drake.mit.edu/pip.html#stable-releases
FROM robotlocomotion/drake:jammy-20230112 AS drake

# essentials
# build-essential and python3-dev are useful sometimes, needed for scikit-sparse
RUN apt-get clean && \
	apt-get update && \
	apt-get -y install gosu && \
	apt-get -y install git && \
	apt-get -y install build-essential && \
	apt-get -y install python3-dev && \
	apt-get -y install gdb

# display and ffmpeg for video writing
RUN apt-get install -qy \
	x11-apps \
	python3-tk \
	xauth \
	ffmpeg \
	wget && \
	apt-get -qy autoremove

# for scikit-sparse
# RUN apt-get -y install libsuitesparse-dev

# install VHACD for convex decomposition of non-convex meshes
COPY docker/testVHACD /usr/bin
RUN chmod +x /usr/bin/testVHACD

# fork for trimesh fixed to allow VHACD 4.0 for decomposing nonconvex objs
RUN pip install -e git+https://github.com/alberthli/trimesh.git#egg=trimesh_fork

# [last updated 08/11/22]
# upgrading python packages in drake base image
WORKDIR /home/user/manipulation
ENV PYTHONPATH "${PYTHONPATH}:/home/user/manipulation"
COPY requirements.txt ./ 
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
	pip install --upgrade -r requirements.txt

# [01/22/23] for "no locator available for file" error when using numba compilation
# see: github.com/numba/numba/issues/4908
ENV NUMBA_CACHE_DIR=/tmp

# entrypoint bash script
# [1] syncs UID and GID with host
# [2] creates matplotlib MPLCONFIGDIR and grants appropriate permissions
# [3] adds pre-commit cache
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
