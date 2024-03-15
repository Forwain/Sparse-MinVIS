import json
import math
import random
import argparse

def datagen(data, anno, unanno):
    selected_frames = {}
    new_annos = []
    videos = data['videos'].copy()
    # video_id: 1, 2, ..., 2238
    n_videos = len(videos)
    unanno_videos = []
    for video in videos:
        unanno_video = video.copy()
        n_frames = math.ceil(len(video['file_names']) * (anno+unanno))
        indexed_file_names = list(enumerate(video['file_names']))
        selected_files = sorted(random.sample(indexed_file_names, k=n_frames))
        split = math.ceil(len(selected_files)*(anno/(anno+unanno)))
        # update videos
        video['file_names'] = [file_name for (i, file_name) in selected_files[:split]]
        video['length'] = split
        unanno_video['id'] = video['id'] + n_videos
        unanno_video['file_names'] = [file_name for (i, file_name) in selected_files[split:]]
        unanno_video['length'] = n_frames - split
        unanno_videos.append(unanno_video)
        selected_frames[video['id']] = selected_files[:split]
        selected_frames[unanno_video['id']] = selected_files[split:]        
    
    for video_id in selected_frames:
        if(video_id > n_videos):
            continue # ignore unannotated frames
        anno_selected_list = [d for d in data['annotations'] if d['video_id'] == video_id]
        for anno_selected in anno_selected_list:
            img_id_selected = [id for (id, name) in selected_frames[video_id]]
            segmentations_selected = [seg for (i, seg) in enumerate(anno_selected['segmentations']) if i in img_id_selected]
            bboxes_selected = [bbox for (i, bbox) in enumerate(anno_selected['bboxes']) if i in img_id_selected]
            areas_selected = [area for (i, area) in enumerate(anno_selected['areas']) if i in img_id_selected]
            anno_selected['segmentations'] = segmentations_selected
            anno_selected['bboxes'] = bboxes_selected
            anno_selected['areas'] = areas_selected
            new_annos.append(anno_selected)

    data['videos'] = videos
    if unanno > 0:
        data['videos'].extend(unanno_videos)
    data['annotations'] = new_annos
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--input", type=str, default="./ytvis_2019/train.json", help="Input file path")
    parser.add_argument("--unanno", type=int, default=40, help="Percentage of frames without annotations")
    parser.add_argument('--anno', type=int, default=10, help="Percentage of frames with annotations")
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()
    if(args.verbose):
        print("Command Line Args:", args)
    
    with open (args.input) as f:
        if(args.verbose):
            print("loading data from {}".format(args.input))
        data = json.load(f)
    if(args.verbose):
        n_videos = len(data['videos'])
        print("{} videos loaded".format(n_videos))
        total_frames = sum(video['length'] for video in data['videos'])
        print("total frames: {}".format(total_frames))
    new_data = datagen(data, 0.01*args.anno, 0.01*args.unanno)
    
    if(args.verbose):
        print("{} videos saved".format(len(new_data['videos'])))
        total_frames = sum(video['length'] for video in new_data['videos'])
        ann_frames = sum(video['length']for video in new_data['videos'] if video['id'] <= n_videos)
        unann_frames = sum(video['length']for video in new_data['videos'] if video['id'] > n_videos)
        print("total frames: {}, annotated:{}, unannotated:{}".format(total_frames, ann_frames, unann_frames) )
        
    with open(args.output, "w") as f_out:
        json.dump(new_data, f_out)