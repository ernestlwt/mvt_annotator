docker run -it --rm --gpus=all \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/wslg:/mnt/wslg \
    -e DISPLAY=$DISPLAY \
    -v /mnt/d/work/data/mvt_data:/data \
    -v /home/ernestlwt/workspace/github/mvt_annotator/:/workspace/mvt_annotator \
mvt bash