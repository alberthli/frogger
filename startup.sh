#!/bin/bash
join=false
while getopts n:j flag
do
	case "${flag}" in
		n) image_name_or_id=${OPTARG};;
		j) join=true;;
	esac
done

# command for linux machines
# counts the number of GPUs, if more than 1, then we activate the nvidia runtime
# see: stackoverflow.com/questions/66611439
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	if [ "$join" = false ] ; then
		if (($(lspci | grep -ci vga) > 1)); then
			docker run --rm -it --net=host --ipc=host -e DISPLAY=$DISPLAY \
			-e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
			-v $HOME/.Xauthority:/root/.Xauthority:rw --device /dev/dri/ \
			-e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) \
			-v $PWD:/home/user/manipulation -v $HOME/.ssh:/home/user/.ssh:ro \
			-v $HOME/.gitconfig:/etc/gitconfig --name=$image_name_or_id \
			--runtime=nvidia --gpus all \
			$image_name_or_id /bin/bash
		else
			docker run --rm -it --net=host --ipc=host -e DISPLAY=$DISPLAY \
			-e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
			-v $HOME/.Xauthority:/root/.Xauthority:rw --device /dev/dri/ \
			-e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) \
			-v $PWD:/home/user/manipulation -v $HOME/.ssh:/home/user/.ssh:ro \
			-v $HOME/.gitconfig:/etc/gitconfig --name=$image_name_or_id \
			$image_name_or_id /bin/bash
		fi
	else
		docker exec -it -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -u user \
		$image_name_or_id /bin/bash
	fi

# command for mac OS
elif [[ "$OSTYPE" == "darwin"* ]]; then
	if [ "$join" = false ] ; then
		if (($(lspci | grep -ci vga) > 1)); then
			IP=$(echo $(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}'))
			xhost + $IP
			docker run --rm -it -p 7000-7010:7000-7010 -p 8080:8080 \
			-e DISPLAY=$IP:0 -e QT_X11_NO_MITSHM=1 \
			-v /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/dri/ \
			-v $HOME/.Xauthority:/root/.Xauthority:rw \
			-e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) \
			-v $PWD:/home/user/manipulation -v $HOME/.ssh:/home/user/.ssh:ro \
			-v $HOME/.gitconfig:/etc/gitconfig --name=$image_name_or_id \
			--runtime=nvidia --gpus all \
			$image_name_or_id /bin/bash
			xhost - $IP
		else
			IP=$(echo $(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}'))
			xhost + $IP
			docker run --rm -it -p 7000-7010:7000-7010 -p 8080:8080 \
			-e DISPLAY=$IP:0 -e QT_X11_NO_MITSHM=1 \
			-v /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/dri/ \
			-v $HOME/.Xauthority:/root/.Xauthority:rw \
			-e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) \
			-v $PWD:/home/user/manipulation -v $HOME/.ssh:/home/user/.ssh:ro \
			-v $HOME/.gitconfig:/etc/gitconfig --name=$image_name_or_id \
			$image_name_or_id /bin/bash
			xhost - $IP
		fi
	else
		IP=$(echo $(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}'))
		xhost + $IP
		docker exec -it -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -u user \
		$image_name_or_id /bin/bash
		xhost - $IP
	fi

else
	echo "Startup script only works for Linux or Mac OS!"
fi