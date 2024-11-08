# mvt_annotator

Multi-View Tracking Annotator aims is a annotation tool to automatically establish reasonable tracking using YOLO and SAM2, with additional UI for labeling the same human along time. 

# Install
Creating a conda environment:
```bash
conda create -n mvt_annotator python=3.11 --file requirements.txt
conda activate mvt_annotator
```

Installing SAM2 on local machine:
```bash
cd segment-anything-2
pip install -e .
```

Downloading SAM2 Checkpoint:
```bash
cd checkpoints ## This would make you go inside /segment-anything-2/checkpoints
./download_ckpts.sh
```
# Instructions

The current MVT annotator handles auto labeling & prompt Re-ID annotation request to the human annotator. It have the following steps:

- **Initialize_YOLO_detection & Initialize_SAM_tracking (Automatic)**: run detector & tracker along the entire video to establish segments of tracking.
- **User_annotation_split/merge_pass (Manual)**: prompt the user with the established tracking segments to let the user split/merge trrackings of the same target.

## Initialize_YOLO_detection
This is a automated pass that will run YOLO along the entire video to detect targets(humans) and save the information for subsequent use. This is a automated pass without human intervention.

## Initialize_SAM_tracking
This pass will use the saved YOLO detections to prompt SAM2 and propagate SAM2's tracking. This is a automated pass without human intervention. The segmented masks will be encoded into a consice format and saved automatically. 

## User_annotation_split_pass
After SAM2, we will have certain number of continuous trackings. However, SAM2 can sometimes make mistakes and have multiple targets in the same tracking. To correct label, we must segment the wrong tracks until each track contains only a single target.

The annotator will be presented one tracking at a time with the options:
- **single target in tracking(next)**: press ```n``` to skip the current segmentation, bacause the tracking is good.
- **two targets in tracking(segment)**: keep clicking 2 of the 10 frames that you believe the segmentation point lies. The annotator will zoom in to your selection to let you refine the segmentation point. If the two clicked frames are adjacent, the segmentation point is confirmed and segmented.

Notice: if multiple segmentation point are desired, always start from the one **early** in time.

- **no target in tracking(delete)**: press ```d``` to delete the tracking.

## User_annotation_merge_pass
SAM2 can track the target within some period, but it cannot Re-ID. This step is for merging different trackings of the same target into one.

The annotator will be presented with one target tracking and N candidate trackings. The annotator need to tell the annotator is the target the same as any of the candidates, and if so, which? Sepecifically, the **left-most column** will be the target and the others are the candidates.

If:
- **no candidate match the target(next)**: press ```n``` to mark negative association between the target and all the candidates.
- **some candidate match the target(match)**: click on the candidate person to merge the tracking.

