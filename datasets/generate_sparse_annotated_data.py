import json
import math
import random
import argparse
import warnings

def datagen_as_new(data, anno, unanno):

    warnings.warn("No longer used", DeprecationWarning)
    # 现在用下面的datagen()直接在原video下保存
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

def datagen(data, anno, unanno):
    selected_frames = {}
    new_annos = []
    videos = data['videos'].copy()
    for video in videos:
        n_frames = math.ceil(len(video['file_names']) * (anno+unanno))
        indexed_file_names = list(enumerate(video['file_names']))
        selected_files = sorted(random.sample(indexed_file_names, k=n_frames)) # ann + unann
        # update videos
        video['file_names'] = [file_name for (i, file_name) in selected_files]
        video['length'] = n_frames
        selected_frames[video['id']] = selected_files
    
    for video_id in selected_frames:
        video_length = len(selected_frames[video_id])
        img_id_selected = [id for (id, name) in selected_frames[video_id]]
        anno_selected_list = [d for d in data['annotations'] if d['video_id'] == video_id]
        anno_frames = set()
        for anno_selected in anno_selected_list:
            anno_frames.update([i for (i, seg) in enumerate(anno_selected['segmentations']) if seg is not None])
        anno_frames = anno_frames & set(img_id_selected) # 保证选到的一定有标注
        split = math.ceil(video_length*(anno/(anno+unanno)))
        select = random.sample(anno_frames, split)
        assert(len(select) > 0), anno_selected_list
        for anno_selected in anno_selected_list:
            anno_selected['segmentations'] = [anno_selected['segmentations'][i] if i in select else None for i in img_id_selected]
            anno_selected['bboxes'] = [anno_selected['bboxes'][i] if i in select else None for i in img_id_selected]
            anno_selected['areas'] = [anno_selected['areas'][i] if i in select else None for i in img_id_selected]
            new_annos.append(anno_selected)

    data['videos'] = videos
    data['annotations'] = new_annos
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--input", type=str, default="./ytvis_2019/train.json", help="Input file path")
    parser.add_argument("--unanno", type=int, default=40, help="Percentage of frames without annotations")
    parser.add_argument('--anno', type=int, default=10, help="Percentage of frames with annotations")
    parser.add_argument('--savenew', action='store_true', help = 'Save unannotated frames as new video')
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
    if args.savenew:
        raise DeprecationWarning("No longer used")
        new_data = datagen_as_new(data, 0.01*args.anno, 0.01*args.unanno)
        if(args.verbose):
            print("{} videos saved".format(len(new_data['videos'])))
            total_frames = sum(video['length'] for video in new_data['videos'])
            ann_frames = sum(video['length']for video in new_data['videos'] if video['id'] <= n_videos)
            unann_frames = sum(video['length']for video in new_data['videos'] if video['id'] > n_videos)
            print("total frames: {}, annotated:{}, unannotated:{}".format(total_frames, ann_frames, unann_frames) )            
    else:
        new_data = datagen(data, 0.01*args.anno, 0.01*args.unanno)
        if(args.verbose):
            print("{} videos saved".format(len(new_data['videos'])))
        
    with open(args.output, "w") as f_out:
        json.dump(new_data, f_out)